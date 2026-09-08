"""Define the image classification model for oil spill detection."""

import torch
import torch.nn as nn
from torchvision import models


def build_classifier():
	"""Build an ImageNet-pretrained ResNet18 for grayscale binary classification."""
	classifier = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

	pretrained_weights = classifier.conv1.weight.detach().mean(dim=1, keepdim=True)
	classifier.conv1 = nn.Conv2d(
		in_channels=1,
		out_channels=64,
		kernel_size=7,
		stride=2,
		padding=3,
		bias=False,
	)
	with torch.no_grad():
		classifier.conv1.weight.copy_(pretrained_weights)

	classifier.fc = nn.Linear(classifier.fc.in_features, 1)
	return classifier
