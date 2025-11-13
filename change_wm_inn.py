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
    degrees = []
    for l in range(max_sh_degree + 1):
        degrees += [l] * (2 * l + 1)
    return torch.tensor(degrees, dtype=torch.float32, device=device)

def degree_weights(max_sh_degree: int, power: float = 1.0) -> torch.Tensor:
    deg_idx = build_sh_degree_index(max_sh_degree)
    if max_sh_degree <= 0:
        w = torch.ones_like(deg_idx)
    else:
        w = (deg_idx / max_sh_degree).pow(power)
        w[deg_idx == 0] = 0.0  # 强保护 DC
    return w


# ------------------ 编码器：输出 SH offset ------------------
class WatermarkSHOffsetEncoder(nn.Module):
    """
    输入: message (1, wm_dim) in {0,1}
    输出: sh_offset_base (sh_dim, 3)，再扩展到 (N, sh_dim, 3)
    注意：sh_dim 应匹配 active_sh_degree 的维度
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
            nn.Tanh()
        )
        self.register_buffer("deg_weight", torch.ones(sh_dim, dtype=torch.float32) if deg_weight is None else deg_weight)

    def forward(self, message: torch.Tensor):
        x = self.net(message)  # (1, sh_dim*3)
        x = x.view(1, self.sh_dim, 3)
        x = x * self.deg_weight.view(1, self.sh_dim, 1)
        x = x * self.alpha
        return x


# ------------------ 解码器（输出logits） ------------------
class WatermarkDecoder(nn.Module):
    def __init__(self, wm_dim=32, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, wm_dim)

    def forward(self, render_img):
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
        if img.dim() == 3:
            img = img.unsqueeze(0)
        out = img
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        if self.brightness > 0:
            b_shift = (torch.randn(out.size(0), 1, 1, 1, device=out.device) * self.brightness)
            out = out + b_shift
        if self.contrast > 0:
            c_scale = 1.0 + (torch.randn(out.size(0), 1, 1, 1, device=out.device) * self.contrast)
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

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # 使用 active_sh_degree 的维度
    active_sh_dim = (gaussian.active_sh_degree + 1) ** 2
    deg_w = degree_weights(gaussian.active_sh_degree, power=args.deg_weight_power)

    # 模型
    encoder = WatermarkSHOffsetEncoder(
        wm_dim=args.message_length, sh_dim=active_sh_dim, alpha=args.wm_alpha, deg_weight=deg_w, hidden=args.encoder_hidden
    ).to(device)
    decoder = WatermarkDecoder(wm_dim=args.message_length).to(device)

    # 参数组学习率
    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": args.enc_lr},
            {"params": decoder.parameters(), "lr": args.dec_lr},
        ]
    )
    bce_with_logits = nn.BCEWithLogitsLoss()
    bce_with_logits_center = nn.BCEWithLogitsLoss()

    aug = SimpleAugment(noise_std=args.aug_noise_std, brightness=args.aug_brightness, contrast=args.aug_contrast).to(device)

    progress_bar = tqdm(range(1, opt.water_iterations + 1), desc="Training progress")

    all_train_cams = scene.getTrainCameras().copy()

    # -------- 单样本过拟合：固定视角 + 两条消息并行训练 --------
    fixed_cam = None
    fixed_message_a = None
    fixed_message_b = None
    if args.overfit_one_sample:
        if len(all_train_cams) == 0:
            all_train_cams = scene.getTrainCameras().copy()
        sel_idx = max(0, min(args.overfit_view_idx, len(all_train_cams) - 1))
        fixed_cam = all_train_cams[sel_idx]
        g = torch.Generator(device=device)
        g.manual_seed(args.overfit_seed)
        fixed_message_a = torch.randint(low=0, high=2, size=(1, args.message_length), generator=g, device=device).float()
        g.manual_seed(args.overfit_seed + 1)
        fixed_message_b = torch.randint(low=0, high=2, size=(1, args.message_length), generator=g, device=device).float()
        if torch.allclose(fixed_message_a, fixed_message_b):
            fixed_message_b = 1.0 - fixed_message_a
        print(f"[Overfit] Using view index {sel_idx}, active_sh_degree={gaussian.active_sh_degree}, active_sh_dim={active_sh_dim}")
        print(f"[Overfit] msgA(first16): {fixed_message_a[0,:16].tolist()}")
        print(f"[Overfit] msgB(first16): {fixed_message_b[0,:16].tolist()}")

    for iteration in range(1, opt.water_iterations + 1):
        # 采样视角
        if args.overfit_one_sample:
            cams = [fixed_cam]
        else:
            if len(all_train_cams) < args.num_views:
                all_train_cams = scene.getTrainCameras().copy()
            cams = []
            for _ in range(args.num_views):
                idx = randint(0, len(all_train_cams) - 1)
                cams.append(all_train_cams.pop(idx))

        total_loss = 0.0
        total_render_loss = 0.0
        total_msg_loss = 0.0
        total_cov = 0.0
        total_ber_a = 0.0
        total_ber_b = 0.0
        total_ber0_a = 0.0
        total_ber0_b = 0.0
        zero_count = 0
        enc_grad_norm = 0.0
        dec_grad_norm = 0.0
        img_delta_l1 = 0.0
        random_img_delta_l1 = 0.0

        # 消息生成
        if args.overfit_one_sample:
            msgA = fixed_message_a
            msgB = fixed_message_b
        else:
            msgA = torch.randint(low=0, high=2, size=(1, args.message_length), device=device).float()
            msgB = torch.randint(low=0, high=2, size=(1, args.message_length), device=device).float()

        # 预先计算两条消息的 offset（与视角无关）
        sh_offset_base_A = encoder(msgA)  # (1, active_sh_dim, 3)
        sh_offset_base_B = encoder(msgB)  # (1, active_sh_dim, 3)
        sh_features_full = gaussian.get_features.detach()  # (N, max_sh_dim, 3)
        sh_features = sh_features_full[:, :active_sh_dim, :]  # (N, active_sh_dim, 3)
        sh_offset_A = sh_offset_base_A.expand(sh_features.shape[0], -1, -1)
        sh_offset_B = sh_offset_base_B.expand(sh_features.shape[0], -1, -1)
        offset_reg = torch.tensor(0.0, device=device)
        if args.lambda_offset > 0.0:
            # sh_offset_base_*: (1, active_sh_dim, 3), deg_w: (active_sh_dim,)
            regA = (sh_offset_base_A.pow(2) * deg_w.view(1, -1, 1)).mean()
            regB = (sh_offset_base_B.pow(2) * deg_w.view(1, -1, 1)).mean()
            offset_reg = 0.5 * (regA + regB)

        optimizer.zero_grad(set_to_none=True)
        for vi, viewpoint_cam in enumerate(cams):
            bg = torch.rand((3), device=device) if opt.random_background else background

            dir_pp = (gaussian.get_xyz.detach() - viewpoint_cam.camera_center.repeat(gaussian.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)

            # base_color
            base_shs_view = (sh_features).transpose(1, 2).contiguous().view(-1, 3, active_sh_dim)
            base_color = eval_sh(gaussian.active_sh_degree, base_shs_view, dir_pp_normalized)
            base_color = torch.clamp(base_color + 0.5, 0.0, 1.0)

            # wm_color for A/B
            shs_view_A = (sh_features + sh_offset_A).transpose(1, 2).contiguous().view(-1, 3, active_sh_dim)
            wm_color_A = eval_sh(gaussian.active_sh_degree, shs_view_A, dir_pp_normalized)
            wm_color_A = torch.clamp(wm_color_A + 0.5, 0.0, 1.0)

            shs_view_B = (sh_features + sh_offset_B).transpose(1, 2).contiguous().view(-1, 3, active_sh_dim)
            wm_color_B = eval_sh(gaussian.active_sh_degree, shs_view_B, dir_pp_normalized)
            wm_color_B = torch.clamp(wm_color_B + 0.5, 0.0, 1.0)

            # 覆盖强度（取A/B平均）
            covA = (wm_color_A - base_color).abs().mean().item()
            covB = (wm_color_B - base_color).abs().mean().item()
            total_cov += 0.5 * (covA + covB)

            # 渲染
            pkgA = render(viewpoint_cam, gaussian, pipe, bg, override_color=wm_color_A)
            imgA = pkgA["render"]
            pkgB = render(viewpoint_cam, gaussian, pipe, bg, override_color=wm_color_B)
            imgB = pkgB["render"]

            # 诊断：图像域差异（仅对A做一次）
            if args.diag_every > 0 and iteration % args.diag_every == 0 and vi == 0:
                with torch.no_grad():
                    pkg_base = render(viewpoint_cam, gaussian, pipe, bg, override_color=base_color)
                    img_base = pkg_base["render"]
                    img_delta_l1 = (imgA - img_base).abs().mean().item()
                    rnd_dir = torch.sign(torch.randn_like(base_color))
                    rnd_color = torch.clamp(base_color + args.diag_override_scale * rnd_dir, 0.0, 1.0)
                    pkg_rnd = render(viewpoint_cam, gaussian, pipe, bg, override_color=rnd_color)
                    img_rnd = pkg_rnd["render"]
                    random_img_delta_l1 = (img_rnd - img_base).abs().mean().item()
                    if iteration % 50 == 0:
                        print(f"[Diag] View {vi}, iteration {iteration}")
                        print(f"img_delta_l1: {img_delta_l1:.5f}, random_img_delta_l1: {random_img_delta_l1:.5f}")

            # 解码（不增强，保持可重复）
            logitsA = decoder(imgA)
            logitsB = decoder(imgB)

            # 损失：overfit 模式仅消息损失，常规模式：重建 + 消息
            # 这里 overfit_one_sample 下，仍然只用消息损失，稳定学习两条消息
            loss_msg_A = bce_with_logits(logitsA, msgA)
            loss_msg_B = bce_with_logits(logitsB, msgB)
            loss_message = loss_msg_A + loss_msg_B

            if args.overfit_one_sample:
                loss_render = torch.tensor(0.0, device=device)
                loss_no_msg = torch.tensor(0.0, device=device)
                if args.no_msg_entropy_w > 0.0:
                    pkg0 = render(viewpoint_cam, gaussian, pipe, bg, override_color=base_color)
                    img0 = pkg0["render"]
                    logits0 = decoder(img0)
                    target05 = torch.full_like(msgA, 0.5)
                    # 对 base 图像同时计算与A/B的“中性”约束（取一次 logits0，分别对比0.5也可）
                    loss_no_msg = bce_with_logits_center(logits0, target05) * args.no_msg_entropy_w
                loss = loss_message + loss_no_msg + args.lambda_offset * offset_reg
            else:
                gt = viewpoint_cam.original_image.to(device)
                loss_render = (1.0 - opt.lambda_dssim) * l1_loss(gt, imgA) + opt.lambda_dssim * (1.0 - ssim(gt, imgA))
                # 常规时还可给 B 分支加同样的重建（这里简化只对A）
                if args.lambda_msg_ramp > 0:
                    ramp = min(1.0, iteration / args.lambda_msg_ramp)
                    lambda_msg_eff = args.lambda_msg * ramp
                else:
                    lambda_msg_eff = args.lambda_msg
                loss = loss_render + lambda_msg_eff * loss_message + args.lambda_offset * offset_reg

            loss.backward()

            # 统计 BER
            with torch.no_grad():
                pA = torch.sigmoid(logitsA); predA = (pA > 0.5).float()
                pB = torch.sigmoid(logitsB); predB = (pB > 0.5).float()
                berA = (predA != msgA).float().mean().item()
                berB = (predB != msgB).float().mean().item()
                total_ber_a += berA
                total_ber_b += berB
                total_loss += loss.item()
                total_msg_loss += (loss_msg_A.item() + loss_msg_B.item())
                total_render_loss += loss_render.item()
                if iteration % 50 == 0:
                    print(f"berA: {berA:.4f}, berB: {berB:.4f}")

            # 零 offset 基线 BER（对A/B都统计一次）
            if args.check_zero_offset_every > 0 and (iteration % args.check_zero_offset_every == 0) and vi == 0:
                with torch.no_grad():
                    pkg0 = render(viewpoint_cam, gaussian, pipe, bg, override_color=base_color)
                    img0 = pkg0["render"]
                    logits0 = decoder(img0)
                    p0 = torch.sigmoid(logits0)
                    pred0 = (p0 > 0.5).float()
                    ber0A = (pred0 != msgA).float().mean().item()
                    ber0B = (pred0 != msgB).float().mean().item()
                    total_ber0_a += ber0A
                    total_ber0_b += ber0B
                    zero_count += 1
                    if iteration % 50 == 0:
                        print(f"ber0A: {ber0A:.4f}, ber0B: {ber0B:.4f}")

            if 0.5 * (covA + covB) > args.cov_warn_threshold and iteration % 50 == 0:
                print(f"[Warn][Iter {iteration}] coverage too high: {(0.5*(covA+covB)):.4f} (> {args.cov_warn_threshold})")

        # 梯度范数
        with torch.no_grad():
            enc_norm = 0.0; dec_norm = 0.0
            for p in encoder.parameters():
                if p.grad is not None:
                    enc_norm += (p.grad.data.norm(2).item()) ** 2
            for p in decoder.parameters():
                if p.grad is not None:
                    dec_norm += (p.grad.data.norm(2).item()) ** 2
            enc_grad_norm = enc_norm ** 0.5
            dec_grad_norm = dec_norm ** 0.5
            if iteration % 50 == 0:
                print(f"enc_grad_norm: {enc_grad_norm:.3e}, dec_grad_norm: {dec_grad_norm:.3e}")

        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            if iteration % 10 == 0:
                k = float(len(cams))
                log_dict = {
                    "Loss": f"{(total_loss/k):.6f}",
                    "L_img": f"{(total_render_loss/k):.6f}",
                    "L_msg": f"{(total_msg_loss/k):.6f}",
                    "BER_A": f"{(total_ber_a/k):.4f}",
                    "BER_B": f"{(total_ber_b/k):.4f}",
                    "cov": f"{(total_cov/k):.5f}",
                    "enc_g": f"{enc_grad_norm:.3e}",
                    "dec_g": f"{dec_grad_norm:.3e}",
                }
                if args.diag_every > 0 and iteration % args.diag_every == 0:
                    log_dict["img_d"] = f"{img_delta_l1:.5f}"
                    log_dict["rnd_img_d"] = f"{random_img_delta_l1:.5f}"
                if zero_count > 0:
                    log_dict["BER0_A"] = f"{(total_ber0_a/max(1,zero_count)):.4f}"
                    log_dict["BER0_B"] = f"{(total_ber0_b/max(1,zero_count)):.4f}"
                progress_bar.set_postfix(log_dict)
                progress_bar.update(10)
                if tb_writer is not None:
                    tb_writer.add_scalar("loss/total", total_loss / k, iteration)
                    tb_writer.add_scalar("loss/render", total_render_loss / k, iteration)
                    tb_writer.add_scalar("loss/message", total_msg_loss / k, iteration)
                    tb_writer.add_scalar("metric/BER_A", total_ber_a / k, iteration)
                    tb_writer.add_scalar("metric/BER_B", total_ber_b / k, iteration)
                    tb_writer.add_scalar("metric/coverage", total_cov / k, iteration)
                    tb_writer.add_scalar("metric/enc_grad_norm", enc_grad_norm, iteration)
                    tb_writer.add_scalar("metric/dec_grad_norm", dec_grad_norm, iteration)
                    if args.diag_every > 0 and iteration % args.diag_every == 0:
                        tb_writer.add_scalar("metric/img_delta_l1", img_delta_l1, iteration)
                        tb_writer.add_scalar("metric/random_img_delta_l1", random_img_delta_l1, iteration)
                    if zero_count > 0:
                        tb_writer.add_scalar("metric/BER0_A", total_ber0_a / max(1, zero_count), iteration)
                        tb_writer.add_scalar("metric/BER0_B", total_ber0_b / max(1, zero_count), iteration)

            if iteration == opt.water_iterations:
                progress_bar.close()

            if (iteration in args.save_iterations):
                print(f"\n[ITER {iteration}] Saving checkpoints")
                torch.save((gaussian.capture(), iteration),
                           os.path.join(scene.model_path, f"chkpnt_{args.message_length}_{iteration}_{args.exp_name}.pth"))
                torch.save(encoder.state_dict(),
                           os.path.join(scene.model_path, f"encoder_{args.message_length}_{iteration}_{args.exp_name}.pth"))
                torch.save(decoder.state_dict(),
                           os.path.join(scene.model_path, f"decoder_{args.message_length}_{iteration}_{args.exp_name}.pth"))

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

    # 新增/调整超参
    parser.add_argument("--exp_name", type=str, default="wm_shoffset")
    parser.add_argument("--lr", type=float, default=1e-4)  # 兼容旧参（不再直接使用）
    parser.add_argument("--enc_lr", type=float, default=2e-4, help="Encoder 学习率")
    parser.add_argument("--dec_lr", type=float, default=1e-4, help="Decoder 学习率")
    parser.add_argument("--wm_alpha", type=float, default=0.02, help="SH offset 全局幅度")
    parser.add_argument("--deg_weight_power", type=float, default=1.5, help="按阶权重指数，>1更强调高阶")
    parser.add_argument("--lambda_msg", type=float, default=1.0, help="消息损失权重（常规训练使用）")
    parser.add_argument("--lambda_msg_ramp", type=int, default=1000, help="消息权重ramp-up步数，0禁用")
    parser.add_argument("--num_views", type=int, default=1, help="每步采样的视角数量（>1提升一致性）")
    parser.add_argument("--use_augment", action="store_true", default=False)
    parser.add_argument("--aug_noise_std", type=float, default=0.01)
    parser.add_argument("--aug_brightness", type=float, default=0.02)
    parser.add_argument("--aug_contrast", type=float, default=0.02)
    parser.add_argument("--encoder_hidden", type=int, default=256)

    # 排查/稳定相关
    parser.add_argument("--overfit_one_sample", action="store_true", default=False,
                        help="单样本过拟合：固定视角，并在两条消息A/B上同步训练")
    parser.add_argument("--overfit_view_idx", type=int, default=0, help="过拟合测试的视角索引")
    parser.add_argument("--overfit_seed", type=int, default=123, help="固定消息的随机种子（A/B使用seed与seed+1）")
    parser.add_argument("--check_zero_offset_every", type=int, default=50,
                        help="每N步以零offset渲染一次做BER基线检查，0不检查")
    parser.add_argument("--cov_warn_threshold", type=float, default=0.05,
                        help="覆盖强度的告警阈值")
    parser.add_argument("--no_msg_entropy_w", type=float, default=0.0,
                        help="零offset图像的中性正则权重（logits→0，对应概率0.5）")
    parser.add_argument("--diag_every", type=int, default=50, help="每N步做一次渲染/图像差诊断")
    parser.add_argument("--diag_override_scale", type=float, default=0.2, help="随机override强度，用于验证renderer响应")
    # Add a CLI arg near other hyperparameters:
    parser.add_argument("--lambda_offset", type=float, default=0.0, help="L2 regularizer on SH offset (per-degree weighted)")

    args = parser.parse_args()
    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), pp.extract(args), args)