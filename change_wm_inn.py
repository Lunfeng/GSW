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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
    """FiLM 风格的编码器，对颜色与消息分别建模后进行逐通道调制"""

    def __init__(self, color_dim=3, wm_dim=32, hidden_dim=128):
        super().__init__()
        self.color_dim = color_dim
        self.wm_dim = wm_dim
        self.base_net = nn.Sequential(
            nn.Linear(color_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, color_dim)
        )
        self.film_gen = nn.Sequential(
            nn.Linear(wm_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, color_dim * 2)
        )

    def forward(self, colors, wm_bits):
        """
        colors: (N,3)
        wm_bits: (wm_dim,) or (1, wm_dim)
        返回扰动后的颜色，形状与 colors 相同
        """
        if wm_bits.dim() == 1:
            wm_bits = wm_bits.unsqueeze(0)
        wm_bits = wm_bits.to(colors.dtype)
        wm_bits_exp = wm_bits.expand(colors.shape[0], -1)
        base_feat = self.base_net(colors)
        gamma_beta = self.film_gen(wm_bits_exp)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        modulation = gamma * base_feat + beta
        residual = torch.tanh(modulation)
        return colors + residual


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
        self.wm_dim = wm_dim
        # 空域分支
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(width, width * 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        spatial_dim = width + width * 2 + width * 4

        # 频域分支
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(width, width * 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.freq_pool = nn.AdaptiveAvgPool2d(1)
        freq_dim = width * 2

        self.spatial_proj = nn.Linear(spatial_dim, 256)
        self.freq_proj = nn.Linear(freq_dim, 256)
        self.attention = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, wm_dim)
        )

    @staticmethod
    def _fft_magnitude(x):
        freq = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        return torch.sqrt(freq.real ** 2 + freq.imag ** 2 + 1e-6)

    def forward(self, x_bchw):
        if x_bchw.dim() == 3:
            x_bchw = x_bchw.unsqueeze(0)

        f1 = self.stem(x_bchw)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        g1 = self.pool(f1).flatten(1)
        g2 = self.pool(f2).flatten(1)
        g3 = self.pool(f3).flatten(1)
        spatial_feat = torch.cat([g1, g2, g3], dim=1)

        freq_input = self._fft_magnitude(x_bchw)
        freq_feat = self.freq_branch(freq_input)
        freq_feat = self.freq_pool(freq_feat).flatten(1)

        spatial_embed = self.spatial_proj(spatial_feat)
        freq_embed = self.freq_proj(freq_feat)
        attn = torch.sigmoid(self.attention(torch.cat([spatial_embed, freq_embed], dim=1)))
        fused = attn * spatial_embed + (1 - attn) * freq_embed
        return self.head(fused)


class ProjectionHead(nn.Module):
    def __init__(self, dim, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def compute_ber_from_logits(wm_logits, true_bits, threshold=0.5, use_sigmoid=True, per_sample=False, tb_writer=None, step=None, progress_bar=None, tb_tag='metrics/BER'):
    if wm_logits.dim() == 1:
        wm_logits = wm_logits.unsqueeze(0)
    if true_bits.dim() == 1:
        true_bits = true_bits.unsqueeze(0)
    if true_bits.shape[0] == 1 and wm_logits.shape[0] > 1:
        true_bits = true_bits.expand(wm_logits.shape[0], -1)
    if true_bits.shape != wm_logits.shape:
        raise ValueError(f"shape mismatch: wm_logits {wm_logits.shape}, true_bits {true_bits.shape}")

    if use_sigmoid:
        preds = (torch.sigmoid(wm_logits) > threshold)
    else:
        preds = (wm_logits > 0.0)
    trues = (true_bits > 0.5)

    diff = (preds != trues).float()
    per_sample_ber = diff.mean(dim=1)
    mean_ber = per_sample_ber.mean().item()

    return per_sample_ber if per_sample else mean_ber


def info_nce_loss(message_bits, pred_logits, msg_projector, pred_projector, temperature=0.07, num_negatives=16):
    """基于 InfoNCE 的互信息最大化目标"""
    if message_bits.dim() == 1:
        message_bits = message_bits.unsqueeze(0)
    negatives = torch.randint(
        0, 2, (num_negatives, message_bits.shape[1]), device=message_bits.device, dtype=message_bits.dtype
    )
    all_messages = torch.cat([message_bits, negatives], dim=0)
    msg_embed = F.normalize(msg_projector(all_messages), dim=-1)
    pred_embed = F.normalize(pred_projector(torch.sigmoid(pred_logits)), dim=-1)
    pos = torch.sum(pred_embed * msg_embed[0:1], dim=-1, keepdim=True)
    neg = pred_embed @ msg_embed[1:].T
    logits = torch.cat([pos, neg], dim=-1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

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
    msg_projector = ProjectionHead(args.message_length).to(device)
    pred_projector = ProjectionHead(args.message_length).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(msg_projector.parameters()) +
        list(pred_projector.parameters()), lr=1e-4
    )
    criterion_BCE = nn.BCEWithLogitsLoss()

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
        message = torch.randint(0, 2, (1, args.message_length), device=device, dtype=torch.float32)

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

        loss_message = criterion_BCE(wm_pred, message)
        loss_info = info_nce_loss(
            message,
            wm_pred,
            msg_projector,
            pred_projector,
            temperature=args.info_temperature,
            num_negatives=args.num_info_negatives
        )
        loss = loss_render + args.lambda_message * loss_message + args.lambda_info * loss_info

        # backward
        loss.backward()

        with torch.no_grad():
            # 在训练循环中（在 `wm_pred = decoder(render_image)` 之后，loss 计算前或后均可），插入如下调用：
            # 计算并输出 BER（示例放在解码后）
            ber = compute_ber_from_logits(wm_pred, message, threshold=0.5, use_sigmoid=True, per_sample=False,
                                          tb_writer=tb_writer, step=iteration, progress_bar=progress_bar)
            # print(f"[ITER {iteration}] BER: {ber:.4f}")
            if iteration % 100 == 0 or iteration == opt.water_iterations:
                target = viewpoint_cam.original_image.cuda()
                pred = render_image
                mse = torch.mean((target - pred) ** 2).item()
                psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float('inf')
                ssim_val = ssim(target, pred)
                if isinstance(ssim_val, torch.Tensor):
                    ssim_val = ssim_val.item()
                print(f"[ITER {iteration}] PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.2f}, BER: {ber:.2f}")
                if tb_writer:
                    tb_writer.add_scalar('metrics/PSNR', psnr, iteration)
                    tb_writer.add_scalar('metrics/SSIM', ssim_val, iteration)
                    tb_writer.add_scalar('metrics/BER', ber, iteration)

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{loss.item():.{3}f}",
                     "l_m": f"{loss_message.item():.{3}f}",
                     "l_i": f"{loss_info.item():.{3}f}",
                     "l_r": f"{loss_render.item():.{3}f}",
                     "BER": f"{ber:.3f}"})
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
    parser.add_argument("--lambda_message", type=float, default=1.0)
    parser.add_argument("--lambda_info", type=float, default=0.1)
    parser.add_argument("--info_temperature", type=float, default=0.07)
    parser.add_argument("--num_info_negatives", type=int, default=32)

    args = parser.parse_args()
    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\nTraining complete.")
