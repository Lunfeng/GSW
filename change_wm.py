import os

import numpy as np
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
#
# os.system('echo running in gpu $CUDA_VISIBLE_DEVICES')
from random import randint

from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from network.Network import *
from utils.dwt_tool import DWTTool, WaveletAnnealingLoss
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim, psnr
import lpips
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans

from utils.sh_utils import eval_sh

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


class ColorWatermarkEncoder(nn.Module):
    def __init__(self, wm_dim=64, color_dim=3, hidden_dim=128, alpha=0.02):
        super().__init__()
        self.alpha = alpha
        self.mlp = nn.Sequential(
            nn.Linear(color_dim + wm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, color_dim)
        )

    def forward(self, colors, wm_bits):
        """
        colors: (N, 3)
        wm_bits: (wm_dim,)
        """
        wm_bits = wm_bits.expand(colors.shape[0], -1)  # (N, wm_dim)
        x = torch.cat([colors, wm_bits], dim=-1)       # (N, 3 + wm_dim)
        c_watermarked = self.mlp(x)                    # (N, 3)
        return c_watermarked


class WatermarkDecoder(nn.Module):
    def __init__(self, wm_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # 将wm_bits融入全连接层
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4 + wm_dim, 512), nn.ReLU(),
            nn.Linear(512, wm_dim),
            nn.Sigmoid()
        )

    def forward(self, img, colors_svd=None):
        """
        img: (B,3,H,W)
        colors_svd: (B, message_length) 高斯核颜色的 SVD 特征
        """
        B = img.shape[0] if img.ndim == 4 else 1
        if img.ndim == 3:
            img = img.unsqueeze(0)

        feat = self.encoder(img)
        feat = feat.flatten(1)  # (B, 256*4*4)

        if colors_svd is not None:
            if colors_svd.ndim == 1:
                colors_svd = colors_svd.unsqueeze(0).expand(B, -1)
            feat = torch.cat([feat, colors_svd], dim=1)

        wm_pred = self.fc(feat)
        return wm_pred

def decoded_message_error_rate(message, decoded_message):
    length = message.shape[0]

    message = message.gt(0.5)
    decoded_message = decoded_message.gt(0.5)
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate

def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate


def training(dataset, opt, pipe, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussian = GaussianModel(dataset.sh_degree)
    scene = Scene(args, gaussian, shuffle=False)
    checkpoint = os.path.join(args.model_path, args.start_checkpoint)
    (model_params, _) = torch.load(checkpoint)
    gaussian.restore(model_params, args)
    gaussian.training_watermark_setup(args)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    print(f'bg_color: {bg_color}')
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    encoder = ColorWatermarkEncoder(wm_dim=args.message_length, alpha=0.02).to(device)
    decoder = WatermarkDecoder(wm_dim=args.message_length).to(device)
    criterion_MSE = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.water_iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.water_iterations + 1):
        iter_start.record()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        shs_view = gaussian.get_features.transpose(1, 2).view(-1, 3, (gaussian.max_sh_degree + 1) ** 2)
        dir_pp = (gaussian.get_xyz - viewpoint_cam.camera_center.repeat(gaussian.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussian.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        message = torch.Tensor(np.random.choice([0, 1], (1, args.message_length))).to(device)

        wm_color = encoder(colors_precomp, message)
        render_pkg = render(viewpoint_cam, gaussian, pipe, bg, override_color=wm_color)

        render_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        print(f'colors_precomp.shape is {colors_precomp.shape}')
        U, S, V = torch.svd(colors_precomp)  # colors_precomp: [N,3]
        print(f'S.shape is {S.shape}')
        colors_svd = S[:args.message_length]  # [message_length]
        colors_svd = colors_svd.to(device)

        wm_pred = decoder(render_image, colors_svd)
        gt_image = viewpoint_cam.original_image.cuda()

        loss_l1 = l1_loss(gt_image, render_image)
        loss_render = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * (1.0 - ssim(gt_image, render_image))

        loss_message = criterion_MSE(message, wm_pred)

        color_loss = criterion_MSE(colors_precomp, wm_color)
        loss = color_loss + loss_message + loss_render
        loss.backward()

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/message_loss", loss_message.item(), iteration)
            tb_writer.add_scalar("Loss/total_loss", loss.item(), iteration)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}",
                                          "loss_message": f"{loss_message.item():.{7}f}",
                                          "color_loss": f"{color_loss.item():.{7}f}",
                                          })
                progress_bar.update(10)
            if iteration == opt.water_iterations:
                progress_bar.close()
            # Log and save
            if (iteration in args.save_iterations):
                # 保存编码器和解码器
                torch.save(encoder.state_dict(),
                           os.path.join(scene.model_path, f"encoder_{iteration}_{args.exp_name}.pth"))
                torch.save(decoder.state_dict(),
                           os.path.join(scene.model_path, f"decoder_{iteration}_{args.exp_name}.pth"))


            if iteration < opt.water_iterations:
                # gaussian.optimizer.step()
                # gaussian.optimizer.zero_grad(set_to_none=True)

                optimizer.step()
                optimizer.zero_grad()

    del encoder, decoder
    torch.cuda.empty_cache()

    with torch.no_grad():
        message = torch.Tensor(np.random.choice([0, 1], (1, args.message_length))).to(device)
        test_encoder = ColorWatermarkEncoder(wm_dim=args.message_length).to(device)
        test_decoder = WatermarkDecoder(wm_dim=args.message_length).to(device)

        test_encoder.load_state_dict(torch.load(os.path.join(args.model_path, f"encoder_{args.water_iterations}_{args.exp_name}.pth")))
        test_decoder.load_state_dict(torch.load(os.path.join(args.model_path, f"decoder_{args.water_iterations}_{args.exp_name}.pth")))
        test_encoder.eval()
        test_decoder.eval()
        views = scene.getTrainCameras().copy()
        errors = []
        psnrs = []
        ssims = []
        lpips_model = lpips.LPIPS(net="vgg").to(device)
        lpipss = []
        for idx, viewpoint_cam in enumerate(tqdm(views, desc="Rendering progress")):

            dir_pp = (gaussian.get_xyz - viewpoint_cam.camera_center.repeat(gaussian.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            shs_view = gaussian.get_features.transpose(1, 2).view(-1, 3, (gaussian.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(gaussian.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            wm_color = test_encoder(colors_precomp, message)
            render_pkg = render(viewpoint_cam, gaussian, pipe, background, override_color=wm_color)

            render_image = render_pkg["render"]
            wm_pred = test_decoder(render_image, wm_bits=message)

            error = decoded_message_error_rate_batch(message, wm_pred)
            errors.append(error)

            gt_image = viewpoint_cam.original_image[0:3, :, :]

            psnrs.append(psnr(gt_image, render_image).mean())
            ssims.append(ssim(gt_image, render_image))
            lpipss.append(lpips_model(gt_image, render_image))
        print(ssims, lpipss, psnrs)

        avg_psnr = float(torch.tensor(psnrs).mean())
        avg_ssim = float(torch.tensor(ssims).mean())
        avg_lpips = float(torch.tensor(lpipss).mean())
        avg_error = (1 - torch.tensor(errors).mean()) * 100


        print(f'exp_name {args.exp_name}, '
              f'avg psnr is {avg_psnr}, '
              f'avg ssim is {avg_ssim}, '
              f'avg lpips is {avg_lpips}'
              f'avg_error is {avg_error}')


if __name__ == "__main__":
    print('--------------------------------Train Gaussian Scene--------------------------------------')
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default="chkpnt30000.pth")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--message_length", type=int, default=32)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.water_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
