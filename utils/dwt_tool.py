import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class DWTTool(nn.Module):
    '''
    工具类：封装DWT分解和IWT合成功能
    '''

    def __init__(self, wave='haar', mode='zero'):
        super(DWTTool, self).__init__()
        self.dwt = DWTForward(J=1, mode=mode, wave=wave)
        self.iwt = DWTInverse(mode=mode, wave=wave)

    def decompose(self, image):
        '''
        DWT分解
        输入: (B, 3, H, W)
        输出: (B, 12, H/2, W/2)
        '''
        yl, yh = self.dwt(image)  # yl: (B, 3, H/2, W/2), yh: [(B, 3, 3, H/2, W/2)]
        yh = yh[0]
        lh, hl, hh = torch.unbind(yh, dim=2)  # (B, 3, H/2, W/2) x3

        dwt_features = torch.cat([yl, lh, hl, hh], dim=1)  # (B, 12, H/2, W/2)
        return dwt_features

    def compose(self, dwt_features):
        '''
        IWT合成
        输入: (B, 12, H/2, W/2)
        输出: (B, 3, H, W)
        '''
        encoded_yl = dwt_features[:, 0:3, :, :]
        encoded_lh = dwt_features[:, 3:6, :, :]
        encoded_hl = dwt_features[:, 6:9, :, :]
        encoded_hh = dwt_features[:, 9:12, :, :]

        encoded_yh = torch.stack([encoded_lh, encoded_hl, encoded_hh], dim=2)  # (B, 3, 3, H/2, W/2)

        reconstructed_image = self.iwt((encoded_yl, [encoded_yh]))
        return reconstructed_image
        # # DWT分解
        # yl, yh = self.dwt(image)  # yl: (B, 3, H/2, W/2), yh: [(B, 3, 2, H/2, W/2)]
        #
        # yh = yh[0]  # 取第一层的高频子带
        # # yh shape: (B, 3, 3, H/2, W/2)  -> 3是LH/HL/HH
        #
        # # 把四个子带拼接
        # # 先把高频展开
        # lh, hl, hh = torch.unbind(yh, dim=2)  # (B, 3, H/2, W/2) x3
        #
        # # cat顺序 [LL, LH, HL, HH]
        # dwt_features = torch.cat([yl, lh, hl, hh], dim=1)  # (B, 12, H/2, W/2)


class WaveletAnnealingLoss(nn.Module):
    def __init__(self, wave='haar', levels=2, start_step=0, end_step=30000):
        """
        wave: 小波类型（如 'haar', 'db1', 'db2' 等）
        levels: 小波分解的最大层数
        start_step: 开始引入高频的迭代数（T0）
        end_step: 完全引入所有高频的迭代数（T）
        """
        super().__init__()
        self.dwt = DWTForward(J=levels, wave=wave)
        self.levels = levels
        self.start_step = start_step
        self.end_step = end_step
        self.mse = nn.MSELoss()

    def _active_levels(self, step):
        """根据当前迭代数 step 计算启用的层数"""
        if step < 10000:
            return 0
        elif 10000 <= step < 20000:
            return 1
        else:
            return 2

    def forward(self, pred, target, step):
        """
        pred: 预测图像 (B, C, H, W)
        target: 真实图像 (B, C, H, W)
        step: 当前迭代数
        """
        # 小波分解
        pred_ll, pred_highs = self.dwt(pred)
        target_ll, target_highs = self.dwt(target)

        loss = self.mse(pred_ll, target_ll)  # 低频分量损失

        active_level = self._active_levels(step)
        for l in range(active_level):
            for band in range(3):  # LH, HL, HH
                pred_band = pred_highs[l][:, band, :, :, :]
                target_band = target_highs[l][:, band, :, :, :]
                loss += self.mse(pred_band, target_band)

        return loss
