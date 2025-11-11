import os
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MBRSDataset(Dataset):
	def __init__(self, path, sort=False):
		super(MBRSDataset, self).__init__()
		self.path = path
		if sort:
			self.list = sorted(os.listdir(path))
		else:
			self.list = os.listdir(path)
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

	def __getitem__(self, index):
		image = Image.open(os.path.join(self.path, self.list[index])).convert("RGB")
		image = self.transform(image)
		if image is not None:
			return image
		# print("dataloader : skip index", index)
		index += 1

	def __len__(self):
		return len(self.list)
