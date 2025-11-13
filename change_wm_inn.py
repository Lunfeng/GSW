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
import random

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


# ------------------ 工具函数 ------------------
def build_sh_degree_index(max_sh_degree: int) -> torch.Tensor:
    # 生成每个基函数对应的阶数 l 索引: [0, 1,1,1, 2,2,2,2,2, ...]
    degrees = []
    for l in range(max_sh_degree + 1):
        degrees += [l] * (2 * l + 1)
    return torch.tensor(degrees, dtype=torch.float32, device=device)

def degree_weights(max_sh_degree: int, power: float = 1.0) -> torch.Tensor:
    # 给高阶更大权重，保护低阶（低阶更影响整体颜色/PSNR）
    deg_idx = build_sh_degree_index(max_sh_degree)  # shape [sh_dim]
    if max_sh_degree <= 0:
        w = torch.ones_like(deg_idx)
    else:
        w = (deg_idx / max_sh_degree).pow(power)  # [0..1]^p
        w[deg_idx == 0] = 0.0  # 强烈保护 DC 项
    return w  # shape [sh_dim]


# ------------------ 编码器：输出 SH offset ------------------
class WatermarkSHOffsetEncoder(nn.Module):
    """
    输入: message (1, wm_dim) in {0,1}
    输出: sh_offset_base (sh_dim, 3), 再扩展到 (N, sh_dim, 3) 用于所有点
    """
    def __init__(self, wm_dim: int, sh_dim: int, alpha: float = 0.01, deg_weight: torch.Tensor = None, hidden: int = 256):
        super().__init__()
        self.wm_dim = wm_dim
        self.sh_dim = sh_dim
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(wm_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, sh_dim * 3),
            nn.Tanh()  # 限幅到 [-1,1]，再乘以 alpha
        )
        self.register_buffer("deg_weight", torch.ones(sh_dim, dtype=torch.float32) if deg_weight is None else deg_weight)

    def forward(self, message: torch.Tensor):
        # message: (1, wm_dim), 值域 {0,1}
        x = self.net(message)  # (1, sh_dim*3)
        x = x.view(1, self.sh_dim, 3)  # (1, sh_dim, 3)
        # 按阶权重：保护低阶，强调高阶
        x = x * self.deg_weight.view(1, self.sh_dim, 1)
        # 缩放幅度
        x = x * self.alpha
        return x  # (1, sh_dim, 3)


# ------------------ 解码器（更稳健，输出logits） ------------------
class WatermarkDecoder(nn.Module):
    def __init__(self, wm_dim=32, in_channels=3):
        super().__init__()
        # 轻量但更稳解码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),  # downsample
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),  # downsample
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, wm_dim)

    def forward(self, render_img):
        """
        render_img: (3,H,W) 或 (1,3,H,W)
        返回: (1, wm_dim) logits
        """
        if render_img.dim() == 3:
            render_img = render_img.unsqueeze(0)
        feats = self.encoder(render_img)
        feats = feats.view(feats.size(0), -1)
        logits = self.fc(feats)
        return logits


# ------------------ 轻量可微增强 ------------------
class SimpleAugment(nn.Module):
    def __init__(self, noise_std=0.01, brightness=0.05, contrast=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, img):
        # img: (1,3,H,W) 或 (3,H,W)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        b, c, h, w = img.shape
        out = img
        # 轻微高斯噪声
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        # 亮度/对比度扰动（保持可微）
        if self.brightness > 0:
            b_shift = (torch.randn(b, 1, 1, 1, device=out.device) * self.brightness)
            out = out + b_shift
        if self.contrast > 0:
            c_scale = 1.0 + (torch.randn(b, 1, 1, 1, device=out.device) * self.contrast)
            mean = out.mean(dim=(2, 3), keepdim=True)
            out = (out - mean) * c_scale + mean
        out = torch.clamp(out, 0.0, 1.0)
        return out


# ------------------ 训练函数 ------------------
def training(dataset, opt, pipe, args):
    tb_writer = prepare_output_and_logger(dataset)
    gaussian = GaussianModel(dataset.sh_degree)
    scene = Scene(args, gaussian, shuffle=False)
    checkpoint = os.path.join(args.model_path, args.start_checkpoint)
    print(f"Loading checkpoint from {checkpoint}")
    (model_params, _) = torch.load(checkpoint, map_location=device)
    gaussian.restore(model_params, args)
    gaussian.training_watermark_setup(args)

    # 背景设置
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # SH维度
    sh_dim = (gaussian.max_sh_degree + 1) ** 2
    deg_w = degree_weights(gaussian.max_sh_degree, power=args.deg_weight_power)

    # 模型
    encoder = WatermarkSHOffsetEncoder(
        wm_dim=args.message_length, sh_dim=sh_dim, alpha=args.wm_alpha, deg_weight=deg_w, hidden=args.encoder_hidden
    ).to(device)
    decoder = WatermarkDecoder(wm_dim=args.message_length).to(device)

    # 优化器与损失
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr
    )
    bce_with_logits = nn.BCEWithLogitsLoss()

    # 数据增强
    aug = SimpleAugment(noise_std=args.aug_noise_std, brightness=args.aug_brightness, contrast=args.aug_contrast).to(device)

    # 训练进度
    progress_bar = tqdm(range(1, opt.water_iterations + 1), desc="Training progress")

    # 提前准备视角列表
    all_train_cams = scene.getTrainCameras().copy()

    for iteration in range(1, opt.water_iterations + 1):
        # 小批量视角
        if len(all_train_cams) < args.num_views:
            all_train_cams = scene.getTrainCameras().copy()

        # 随机取 K 个不同视角
        cams = []
        for _ in range(args.num_views):
            idx = randint(0, len(all_train_cams) - 1)
            cams.append(all_train_cams.pop(idx))

        total_loss = 0.0
        total_render_loss = 0.0
        total_msg_loss = 0.0
        total_ber = 0.0

        # 随机生成水印，比特{0,1}
        message = torch.randint(low=0, high=2, size=(1, args.message_length), device=device).float()

        # 由水印 -> SH offset（全局），广播到所有点
        sh_offset_base = encoder(message)  # (1, sh_dim, 3)
        # 不让消息损失反向传播到原 SH（保护 PSNR），只使用其值参与合成颜色
        sh_features = gaussian.get_features.detach()  # (N, sh_dim, 3)
        # 广播 offset 到所有点
        sh_offset = sh_offset_base.expand(sh_features.shape[0], -1, -1)  # (N, sh_dim, 3)

        # 多视角累积
        optimizer.zero_grad(set_to_none=True)
        for viewpoint_cam in cams:
            bg = torch.rand((3), device=device) if opt.random_background else background

            # 计算该视角方向向量
            dir_pp = (gaussian.get_xyz.detach() - viewpoint_cam.camera_center.repeat(gaussian.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)

            # 用(原SH + offset)合成视角颜色
            # get_features: (N, sh_dim, 3) -> (N,3,sh_dim)
            shs_view = (sh_features + sh_offset).transpose(1, 2).contiguous().view(-1, 3, sh_dim)
            wm_color = eval_sh(gaussian.active_sh_degree, shs_view, dir_pp_normalized)
            # 贴近渲染器色域：3DGS 通常是 [-0.5, +0.5] + 0.5 或类似，这里与原代码保持一致
            wm_color = torch.clamp(wm_color + 0.5, 0.0, 1.0)

            # 用 override_color 渲染
            render_pkg = render(viewpoint_cam, gaussian, pipe, bg, override_color=wm_color)
            render_image = render_pkg["render"]  # 期望 (3,H,W) 且在 [0,1]

            # 轻量鲁棒增强
            if args.use_augment:
                render_image_for_dec = aug(render_image).squeeze(0)
            else:
                render_image_for_dec = render_image

            # 解码（输出logits）
            wm_logits = decoder(render_image_for_dec)

            # 损失
            # - 图像重建: 与GT贴合（保护PSNR/SSIM）
            gt = viewpoint_cam.original_image.to(device)
            loss_render = (1.0 - opt.lambda_dssim) * l1_loss(gt, render_image) + \
                          opt.lambda_dssim * (1.0 - ssim(gt, render_image))

            # - 消息: BCEWithLogitsLoss
            loss_message = bce_with_logits(wm_logits, message)

            # 权重随训练进程ramp-up（先稳图像，再拉BER）
            if args.lambda_msg_ramp > 0:
                ramp = min(1.0, iteration / args.lambda_msg_ramp)
                lambda_msg_eff = args.lambda_msg * ramp
            else:
                lambda_msg_eff = args.lambda_msg

            loss = loss_render + lambda_msg_eff * loss_message

            # 反向传播（累积）
            loss.backward()

            # BER统计（不参与反传）
            with torch.no_grad():
                probs = torch.sigmoid(wm_logits)
                pred_bits = (probs > 0.5).float()
                ber = (pred_bits != message).float().mean().item()
                total_ber += ber
                total_loss += loss.item()
                total_render_loss += loss_render.item()
                total_msg_loss += loss_message.item()

        # 更新
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            if iteration % 10 == 0:
                k = float(args.num_views)
                progress_bar.set_postfix({
                    "Loss": f"{(total_loss/k):.6f}",
                    "L_img": f"{(total_render_loss/k):.6f}",
                    "L_msg": f"{(total_msg_loss/k):.6f}",
                    "BER": f"{(total_ber/k):.4f}"
                })
                progress_bar.update(10)
                if tb_writer is not None:
                    tb_writer.add_scalar("loss/total", total_loss / k, iteration)
                    tb_writer.add_scalar("loss/render", total_render_loss / k, iteration)
                    tb_writer.add_scalar("loss/message", total_msg_loss / k, iteration)
                    tb_writer.add_scalar("metric/BER", total_ber / k, iteration)
                    tb_writer.add_scalar("hyper/lambda_msg_eff", lambda_msg_eff, iteration)

            if iteration == opt.water_iterations:
                progress_bar.close()

            # 保存（这里只保存编码器/解码器；真正把 offset 烘入模型建议用第二阶段流程）
            if (iteration in args.save_iterations):
                print(f"\n[ITER {iteration}] Saving checkpoints")
                # 保存当前 3DGS 快照（未改动SH，保持一致）
                torch.save((gaussian.capture(), iteration),
                           os.path.join(scene.model_path, f"chkpnt_{args.message_length}_{iteration}_{args.exp_name}.pth"))
                # 保存编码器/解码器
                torch.save(encoder.state_dict(),
                           os.path.join(scene.model_path, f"encoder_{args.message_length}_{iteration}_{args.exp_name}.pth"))
                torch.save(decoder.state_dict(),
                           os.path.join(scene.model_path, f"decoder_{args.message_length}_{iteration}_{args.exp_name}.pth"))

    # 训练完成
    print("\nTraining complete.")


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

    # 新增超参
    parser.add_argument("--exp_name", type=str, default="wm_shoffset")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wm_alpha", type=float, default=0.02, help="SH offset全局幅度（小幅保障PSNR）")
    parser.add_argument("--deg_weight_power", type=float, default=1.0, help="按阶权重指数，>1更强调高阶")
    parser.add_argument("--lambda_msg", type=float, default=1.0, help="消息损失权重基值")
    parser.add_argument("--lambda_msg_ramp", type=int, default=1000, help="消息权重ramp-up步数，0表示禁用")
    parser.add_argument("--num_views", type=int, default=1, help="每步采样的视角数量（>1可提升跨视角一致性）")
    parser.add_argument("--use_augment", action="store_true", default=True)
    parser.add_argument("--aug_noise_std", type=float, default=0.01)
    parser.add_argument("--aug_brightness", type=float, default=0.05)
    parser.add_argument("--aug_contrast", type=float, default=0.05)
    parser.add_argument("--encoder_hidden", type=int, default=256)

    args = parser.parse_args()
    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), pp.extract(args), args)