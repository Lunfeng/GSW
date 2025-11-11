import torch
import torch.nn as nn
import kornia.augmentation as K
from kornia.augmentation import AugmentationBase2D
from torchvision import transforms
from augly.image import functional as aug_functional
import numpy as np

image_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unnormalize_img(x):
    """ Unnormalize image to [0,1] """
    return (x * image_std.to(x.device)) + image_mean.to(x.device)


def normalize_img(x):
    """ Normalize image to approx. [-1,1] """
    return (x - image_mean.to(x.device)) / image_std.to(x.device)


def clamp_pixel(x):
    """
    Clamp pixel values to 0 255.
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    x_pixel = 255 * unnormalize_img(x)
    y = x_pixel.clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: Tensor image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    x = unnormalize_img(x)
    for ii,img in enumerate(x):
        pil_img = to_pil(img)
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)

class DiffJPEG(nn.Module):
    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality

    def forward(self, x):
        with torch.no_grad():
            img_clip = clamp_pixel(x)
            img_jpeg = jpeg_compress(img_clip, self.quality)
            img_gap = img_jpeg - x
            img_gap = img_gap.detach()
        img_aug = x + img_gap
        return img_aug


class RandomDiffJPEG(AugmentationBase2D):
    def __init__(self, p, low=10, high=100) -> None:
        super().__init__(p=p)
        self.diff_jpegs = [DiffJPEG(quality=qf).to(device) for qf in range(low,high,10)]

    def generate_parameters(self, input_shape: torch.Size):
        qf = torch.randint(high=len(self.diff_jpegs), size=input_shape[0:1])
        return dict(qf=qf)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        qf = params['qf']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.diff_jpegs[qf[ii]](input[ii:ii+1])
        return output

class RandomBlur(AugmentationBase2D):
    def __init__(self, blur_size, p=1) -> None:
        super().__init__(p=p)
        self.gaussian_blurs = [K.RandomGaussianBlur(kernel_size=(kk,kk), sigma= (kk*0.15 + 0.35, kk*0.15 + 0.35)) for kk in range(1,int(blur_size),2)]

    def generate_parameters(self, input_shape: torch.Size):
        blur_strength = torch.randint(high=len(self.gaussian_blurs), size=input_shape[0:1])
        return dict(blur_strength=blur_strength)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        blur_strength = params['blur_strength']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.gaussian_blurs[blur_strength[ii]](input[ii:ii+1])
        return output

class RandomResize(AugmentationBase2D):
    def __init__(self, H, W, min_scale, max_scale=1.0, p=1) -> None:
        super().__init__(p=p)
        self.resizes = [
            K.AugmentationSequential(
                K.Resize((int(H * (ratio / 10)), int(W * (ratio / 10)))),
                K.Resize((H, W)))
            for ratio in range(int(min_scale * 10), int(max_scale * 10))
        ]

    def generate_parameters(self, input_shape: torch.Size):
        ratio = torch.randint(high=len(self.resizes), size=input_shape[0:1])
        return dict(ratio=ratio)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        ratio = params['ratio']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.resizes[ratio[ii]](input[ii:ii+1])
        return output

class Identity(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, noised_and_cover):
        return noised_and_cover


class KorniaAug(nn.Module):
    def __init__(self,
                 H = 750,
                 W = 1000,
                 degrees=45,
                 p_aff=1.0,
                 crop_scale=(0.5, 1.0),
                 crop_ratio=(3 / 4, 3 / 4),
                 cropping_mode='slice',
                 p_crop=1.0,
                 resize_ratio=0.5,
                 p_resize=1.0,
                 blur_size=11,
                 p_blur=1.0,
                 diff_jpeg=30,
                 p_diff_jpeg=1.0,
                 ):
        super(KorniaAug, self).__init__()
        self.aff = K.RandomAffine(degrees=degrees, p=p_aff).to(device)
        self.crop = K.RandomResizedCrop(size=(H, W), scale=crop_scale, ratio=crop_ratio, p=p_crop,
                                        cropping_mode=cropping_mode).to(device)
        self.resize = RandomResize(H, W, resize_ratio, p=p_resize).to(device)
        # self.hflip = K.RandomHorizontalFlip().to(device)
        self.gaussian_blur = RandomBlur(blur_size=blur_size, p=p_blur).to(device)
        self.diff_jpeg = RandomDiffJPEG(p=p_diff_jpeg, low=diff_jpeg).to(device)
        self.gaussian_noise = K.RandomGaussianNoise(mean=0.0, std=1.0, p=1.0).to(device)

        self.noise_layers = [Identity()]
        self.noise_layers.append(self.aff)
        self.noise_layers.append(self.crop)
        self.noise_layers.append(self.resize)
        self.noise_layers.append(self.gaussian_blur)
        # self.noise_layers.append(self.diff_jpeg)
        self.noise_layers.append(self.gaussian_noise)


    def forward(self, input):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        # print(f'random_noise_layer is {random_noise_layer}')
        return random_noise_layer(input)

