import glob
from PIL import Image
from natsort import natsorted
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import config as c
import os

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class DNN_Dataset(Dataset):
    def __init__(self, transforms, mode="train"):
        self.transform = transforms
        self.mode = mode
        if mode == 'train':
            train_path = os.path.join(c.TRAIN_PATH, f"*.{c.format_train}")
            self.files = natsorted(glob.glob(train_path))
        else:
            val_path = os.path.join(c.VAL_PATH, f"*.{c.format_val}")
            print(f'val_path is {val_path}')
            self.files = natsorted(glob.glob(val_path))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item
        except Exception as e:
            print(f"Error loading image {self.files[index]} at index {index}: {e}. Retrying...")
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)

transform = T.Compose([
    T.ToTensor(),
    # T.Resize((256, 256)),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

trainloader = DataLoader(
    DNN_Dataset(transforms=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)
