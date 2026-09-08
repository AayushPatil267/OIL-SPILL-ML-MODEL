"""Define the semantic segmentation model for oil spill boundaries."""

import segmentation_models_pytorch as smp


def build_segmentation_model():
	"""Build an ImageNet-pretrained ResNet34 U-Net for grayscale segmentation."""
	return smp.Unet(
		encoder_name="resnet34",
		encoder_weights="imagenet",
		in_channels=1,
		classes=1,
		activation=None,
	)
