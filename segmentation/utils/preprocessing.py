"""Provide image loading, normalization, and augmentation helpers."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_segmentation_data(images_dir, masks_dir, size=256):
	"""Load paired grayscale images and binary masks from two pre-split folders."""
	image_directory = Path(images_dir)
	mask_directory = Path(masks_dir)
	images = []
	masks = []

	if not image_directory.is_dir() or not mask_directory.is_dir():
		print("Warning: image or mask directory does not exist.")
		print("Total loaded segmentation pairs: 0")
		return images, masks

	for image_path in sorted(image_directory.iterdir()):
		if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
			continue

		mask_path = mask_directory / image_path.name
		if not mask_path.is_file():
			print(f"Warning: skipping {image_path.name}; matching mask not found.")
			continue

		image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
		mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
		if image is None or mask is None:
			print(f"Warning: skipping {image_path.name}; image or mask could not be read.")
			continue

		image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
		mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
		images.append(image.astype(np.float32) / 255.0)
		masks.append((mask >= 127).astype(np.float32))

	print(f"Total loaded segmentation pairs: {len(images)}")
	return images, masks


class SegmentationDataset(Dataset):
	"""Return paired image and binary mask tensors for segmentation training."""

	def __init__(self, images, masks):
		if len(images) != len(masks):
			raise ValueError("images and masks must contain the same number of samples")
		self.images = images
		self.masks = masks

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		image = torch.from_numpy(np.ascontiguousarray(self.images[index])).float().unsqueeze(0)
		mask = torch.from_numpy(np.ascontiguousarray(self.masks[index])).float().unsqueeze(0)
		return image, mask
