import os
import numpy as np
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import uuid

# ------------------ GPU选择 ------------------
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
# os.system('echo running in gpu $CUDA_VISIBLE_DEVICES')

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# ------------------ 自己的库 ------------------
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import eval_sh

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ------------------ 输出和日志 ------------------
def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


# ------------------ 编码器 ------------------
class WatermarkEncoder(nn.Module):
    def __init__(self, color_dim=3, wm_dim=32, hidden_dim=128):
        super().__init__()
        self.color_dim = color_dim
        self.wm_dim = wm_dim
        self.net = nn.Sequential(
            nn.Linear(color_dim + wm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, color_dim),
            nn.Tanh()  # 限制输出颜色变化范围 [-1,1]
        )

    def forward(self, colors, wm_bits):
        """
        colors: (N,3)
        wm_bits: (wm_dim,) or (1, wm_dim)
        """
        wm_bits_exp = wm_bits.expand(colors.shape[0], -1)
        x = torch.cat([colors, wm_bits_exp], dim=1)
        wm_colors = self.net(x)
        return wm_colors


# ------------------ 解码器 ------------------
# class WatermarkDecoder(nn.Module):
#     def __init__(self, wm_dim=32):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1)  # H*W -> 1*1
#         )
#         self.fc = nn.Linear(64, wm_dim)
#
#     def forward(self, render_img):
#         """
#         render_img: (B,3,H,W)
#         returns: (B, wm_dim)
#         """
#         render_img = render_img.unsqueeze(0)
#         features = self.encoder(render_img)
#         features = features.view(features.size(0), -1)
#         wm_pred = self.fc(features)
#         return wm_pred

import torch
import torch.nn as nn

class WatermarkDecoder(nn.Module):
    def __init__(self, wm_dim=32, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(width, width*2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width*2, width*2, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(width*2, width*4, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width*4, width*4, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(width + width*2 + width*4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, wm_dim)
        )

    def forward(self, x_bchw):
        # x: (B,3,H,W)
        x_bchw = x_bchw.unsqueeze(0)
        f1 = self.stem(x_bchw)      # (B,w,H,W)
        f2 = self.down1(f1)         # (B,2w,H/2,W/2)
        f3 = self.down2(f2)         # (B,4w,H/4,W/4)
        g1 = self.pool(f1).flatten(1)
        g2 = self.pool(f2).flatten(1)
        g3 = self.pool(f3).flatten(1)
        g  = torch.cat([g1,g2,g3], dim=1)
        return self.head(g)         # (B, wm_dim)


# ------------------ 训练函数 ------------------
def training(dataset, opt, pipe, args):
    tb_writer = prepare_output_and_logger(dataset)
    gaussian = GaussianModel(dataset.sh_degree)
    scene = Scene(args, gaussian, shuffle=False)
    checkpoint = os.path.join(args.model_path, args.start_checkpoint)
    print(f"Loading checkpoint from {checkpoint}")
    (model_params, _) = torch.load(checkpoint)
    gaussian.restore(model_params, args)
    gaussian.training_watermark_setup(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    encoder = WatermarkEncoder(color_dim=3, wm_dim=args.message_length).to(device)
    decoder = WatermarkDecoder(wm_dim=args.message_length).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4
    )
    criterion_MSE = nn.MSELoss()

    viewpoint_stack = None
    progress_bar = tqdm(range(1, opt.water_iterations + 1), desc="Training progress")

    for iteration in range(1, opt.water_iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        bg = torch.rand((3), device=device) if opt.random_background else background

        shs_view = gaussian.get_features.transpose(1, 2).view(-1, 3, (gaussian.max_sh_degree + 1) ** 2)
        dir_pp = (gaussian.get_xyz - viewpoint_cam.camera_center.repeat(gaussian.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussian.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        # 随机生成水印比特
        message = torch.Tensor(np.random.choice([0, 1], (1, args.message_length))).to(device)

        # 编码
        wm_color = encoder(colors_precomp, message)
        # 渲染
        render_pkg = render(viewpoint_cam, gaussian, pipe, bg, override_color=wm_color)
        render_image = render_pkg["render"]
        # 解码
        wm_pred = decoder(render_image)

        # 损失
        loss_render = (1.0 - opt.lambda_dssim) * l1_loss(viewpoint_cam.original_image.cuda(), render_image) + \
                      opt.lambda_dssim * (1.0 - ssim(viewpoint_cam.original_image.cuda(), render_image))

        loss_message = criterion_MSE(message, wm_pred)
        loss = loss_render + loss_message

        # backward
        loss.backward()

        with torch.no_grad():
            if iteration % 100 == 0 or iteration == opt.water_iterations:
                target = viewpoint_cam.original_image.cuda()
                pred = render_image
                mse = torch.mean((target - pred) ** 2).item()
                psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float('inf')
                ssim_val = ssim(target, pred)
                if isinstance(ssim_val, torch.Tensor):
                    ssim_val = ssim_val.item()
                print(f"[ITER {iteration}] PSNR: {psnr:.4f} dB, SSIM: {ssim_val:.4f}")
                if tb_writer:
                    tb_writer.add_scalar('metrics/PSNR', psnr, iteration)
                    tb_writer.add_scalar('metrics/SSIM', ssim_val, iteration)

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{loss.item():.{7}f}", "loss_message": f"{loss_message.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.water_iterations:
                progress_bar.close()
            # Log and save
            if (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                torch.save((gaussian.capture(), iteration),
                           scene.model_path + f"/chkpnt_{args.message_length}_{iteration}_{args.exp_name}.pth")

                # 保存编码器和解码器
                torch.save(encoder.state_dict(),
                           os.path.join(scene.model_path, f"encoder_{iteration}_{args.exp_name}.pth"))
                torch.save(decoder.state_dict(),
                           os.path.join(scene.model_path, f"decoder_{iteration}_{args.exp_name}.pth"))
            if iteration < opt.water_iterations:
                gaussian.optimizer.step()
                gaussian.optimizer.zero_grad(set_to_none=True)

                optimizer.step()
                optimizer.zero_grad()

# ------------------ 主函数 ------------------
if __name__ == "__main__":
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
    parser.add_argument("--message_length", type=int, default=32)

    args = parser.parse_args()
    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\nTraining complete.")
