import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

from .Encoder_MP import Encoder_MP_Diffusion
from .Decoder import Decoder_Diffusion
from .Noise import Noise
from utils.dwt_tool import DWTTool


class EncoderDecoder_Diffusion(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder (with pytorch_wavelets)
    '''

    def __init__(self, H, W, message_length, noise_layers):
        super(EncoderDecoder_Diffusion, self).__init__()
        # 使用 pytorch_wavelets 的 DWT/IWT
        self.dwt_tool = DWTTool(wave='haar', mode='zero')

        # 传入的是四个子带拼接后（3*4=12通道）
        self.encoder = Encoder_MP_Diffusion(int(H / 2), int(W / 2), message_length)
        self.noise = Noise(noise_layers)
        self.decoder = Decoder_Diffusion(int(H / 2), int(W / 2), message_length)

    def forward(self, image, message):
        dwt_features = self.dwt_tool.decompose(image)
        # Encoder编码
        encoded_features = self.encoder(dwt_features, message)  # 输出 shape 应该是 (B, 12, H/2, W/2)
        encoded_image = self.dwt_tool.compose(encoded_features)

        # 加噪声
        noised_image = self.noise([encoded_image, image])

        # 再做一次 DWT 分解
        noised_dwt_features = self.dwt_tool.decompose(noised_image)
        # 解码
        decoded_message = self.decoder(noised_dwt_features)

        return encoded_image, noised_image, decoded_message
# import torch
# import torch.nn as nn
# from pytorch_wavelets import DWTForward, DWTInverse
#
# from .Encoder_MP import Encoder_MP_Diffusion
# from .Decoder import Decoder_Diffusion
# from .Noise import Noise

# class EncoderDecoder_Diffusion(nn.Module):
#     '''
#     A Sequential of Encoder_MP-Noise-Decoder (with pytorch_wavelets)
#     '''
#
#     def __init__(self, H, W, message_length, noise_layers):
#         super(EncoderDecoder_Diffusion, self).__init__()
#         # 使用 pytorch_wavelets 的 DWT/IWT
#         self.dwt = DWTForward(J=1, mode='zero', wave='haar')
#         self.iwt = DWTInverse(mode='zero', wave='haar')
#
#         # 传入的是四个子带拼接后（3*4=12通道）
#         self.encoder = Encoder_MP_Diffusion(int(H/2), int(W/2), message_length)
#         self.noise = Noise(noise_layers)
#         self.decoder = Decoder_Diffusion(int(H/2), int(W/2), message_length)
#
#     def forward(self, image, message):
#         # DWT分解
#         yl, yh = self.dwt(image)  # yl: (B, 3, H/2, W/2), yh: [(B, 3, 2, H/2, W/2)]
#
#         yh = yh[0]  # 取第一层的高频子带
#         # yh shape: (B, 3, 3, H/2, W/2)  -> 3是LH/HL/HH
#
#         # 把四个子带拼接
#         # 先把高频展开
#         lh, hl, hh = torch.unbind(yh, dim=2)  # (B, 3, H/2, W/2) x3
#
#         # cat顺序 [LL, LH, HL, HH]
#         dwt_features = torch.cat([yl, lh, hl, hh], dim=1)  # (B, 12, H/2, W/2)
#
#         # Encoder编码
#         encoded_features = self.encoder(dwt_features, message)  # 输出 shape 应该是 (B, 12, H/2, W/2)
#
#         # 还原：拆分回来
#         encoded_yl = encoded_features[:, 0:3, :, :]    # LL
#         encoded_lh = encoded_features[:, 3:6, :, :]    # LH
#         encoded_hl = encoded_features[:, 6:9, :, :]    # HL
#         encoded_hh = encoded_features[:, 9:12, :, :]   # HH
#
#         # 把新的高频打包
#         encoded_yh = torch.stack([encoded_lh, encoded_hl, encoded_hh], dim=2)  # (B, 3, 3, H/2, W/2)
#
#         # IWT逆变换
#         encoded_image = self.iwt((encoded_yl, [encoded_yh]))  # 注意 iwt需要 (低频, [高频])
#
#         # 加噪声
#         noised_image = self.noise([encoded_image, image])
#
#         # 再做一次 DWT 分解
#         noised_yl, noised_yh = self.dwt(noised_image)
#         noised_yh = noised_yh[0]
#         noised_lh, noised_hl, noised_hh = torch.unbind(noised_yh, dim=2)
#
#         # 拼接 noised 图像的子带
#         noised_dwt_features = torch.cat([noised_yl, noised_lh, noised_hl, noised_hh], dim=1)  # (B, 12, H/2, W/2)
#
#         # 解码
#         decoded_message = self.decoder(noised_dwt_features)
#
#         return encoded_image, noised_image, decoded_message

# from . import *
# from .Encoder_MP import Encoder_MP_Diffusion
# from .Decoder import Decoder_Diffusion
# from .Noise import Noise
# from utils.dwt_common import *
#
#
# class EncoderDecoder_Diffusion(nn.Module):
# 	'''
# 	A Sequential of Encoder_MP-Noise-Decoder
# 	'''
#
# 	def __init__(self, H, W, message_length, noise_layers):
# 		super(EncoderDecoder_Diffusion, self).__init__()
# 		self.dwt = DWT()
# 		self.iwt = IWT()
# 		self.encoder = Encoder_MP_Diffusion(int(H / 2), int(W / 2), message_length)
# 		self.noise = Noise(noise_layers)
# 		self.decoder = Decoder_Diffusion(int(H / 2), int(W / 2), message_length)
#
# 	def forward(self, image, message):
# 		image_dwt = self.dwt(image)
# 		encoded_image = self.encoder(image_dwt, message)
# 		encoded_image_iwt = self.iwt(encoded_image)
# 		noised_image = self.noise([encoded_image_iwt, image])
# 		noised_image_dwt = self.dwt(noised_image)
# 		decoded_message = self.decoder(noised_image_dwt)
# 		return encoded_image_iwt, noised_image, decoded_message

# from . import *
# from .Encoder_MP import Encoder_MP_Diffusion
# from .Decoder import Decoder_Diffusion
# from .Noise import Noise
# from utils.dwt_common import *
#
#
# class EncoderDecoder_Diffusion(nn.Module):
# 	'''
# 	A Sequential of Encoder_MP-Noise-Decoder
# 	'''
#
# 	def __init__(self, H, W, message_length, noise_layers):
# 		super(EncoderDecoder_Diffusion, self).__init__()
# 		self.dwt = DWT()
# 		self.iwt = IWT()
# 		self.encoder = Encoder_MP_Diffusion(H, W, message_length)
# 		self.noise = Noise(noise_layers)
# 		self.decoder = Decoder_Diffusion(H, W, message_length)
#
# 	def forward(self, image, message):
# 		image_dwt = self.dwt(image)
# 		encoded_image = self.encoder(image, message)
# 		noised_image = self.noise([encoded_image, image])
# 		decoded_message = self.decoder(noised_image)
#
# 		return encoded_image, noised_image, decoded_message
