# import torch.nn as nn
# import torch.nn.functional as F
# import torch
#
# class Resize(nn.Module):
#     """
#     Resize the image. The target size is original size * resize_ratio
#     """
#     def __init__(self, resize_ratio_range, interpolation_method='nearest'):
#         super(Resize, self).__init__()
#         self.resize_ratio_min = resize_ratio_range[0]
#         self.resize_ratio_max = resize_ratio_range[1]
#         self.interpolation_method = interpolation_method
#
#
#     def forward(self, noised_and_cover):
#
#         resize_ratio = torch.empty(1).uniform_(self.resize_ratio_min, self.resize_ratio_max).item()
#         image, cover_image = noised_and_cover
#         image = F.interpolate(
#                                     image,
#                                     scale_factor=(resize_ratio, resize_ratio),
#                                     mode=self.interpolation_method)
#
#         return image


import torch.nn as nn
import torch.nn.functional as F

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio = resize_ratio
        self.interpolation_method = interpolation_method


    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        # resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        image = F.interpolate(
                                    image,
                                    scale_factor=(self.resize_ratio, self.resize_ratio),
                                    mode=self.interpolation_method)

        return image
