"""Train and evaluate the oil spill segmentation model."""

from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from segmentation.model import build_segmentation_model
from utils.preprocessing import SegmentationDataset, load_segmentation_data


IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-4
DEBUG = False
DEBUG_SAMPLE_COUNT = 50
DEBUG_EPOCHS = 2


def combined_loss(logits, masks, dice_loss, bce_loss):
	"""Return the sum of Dice loss and binary cross-entropy loss."""
	return dice_loss(logits, masks) + bce_loss(logits, masks)


def run_epoch(model, loader, dice_loss, bce_loss, device, optimizer=None):
	"""Run one training or validation epoch and return loss and IoU."""
	training = optimizer is not None
	model.train(training)
	total_loss = 0.0
	total_intersection = 0
	total_union = 0
	total_samples = 0

	context = torch.enable_grad() if training else torch.no_grad()
	with context:
		for images, masks in loader:
			images = images.to(device)
			masks = masks.to(device)

			if training:
				optimizer.zero_grad()

			logits = model(images)
			loss = combined_loss(logits, masks, dice_loss, bce_loss)

			if training:
				loss.backward()
				optimizer.step()

			predictions = torch.sigmoid(logits) >= 0.5
			target_masks = masks >= 0.5
			total_intersection += (predictions & target_masks).sum().item()
			total_union += (predictions | target_masks).sum().item()
			total_loss += loss.item() * images.size(0)
			total_samples += images.size(0)

	mean_loss = total_loss / total_samples
	iou = total_intersection / total_union if total_union else 1.0
	return mean_loss, iou


def main():
	"""Train the segmentation model and save the checkpoint with the best IoU."""
	project_dir = Path(__file__).resolve().parents[1]
	train_images_dir = project_dir / "data" / "images" / "train"
	train_masks_dir = project_dir / "data" / "masks" / "train"
	val_images_dir = project_dir / "data" / "images" / "val"
	val_masks_dir = project_dir / "data" / "masks" / "val"
	checkpoint_path = project_dir / "segmentation" / "oil_spill_segmentation.pth"

	train_images, train_masks = load_segmentation_data(
		train_images_dir,
		train_masks_dir,
		size=IMAGE_SIZE,
	)
	val_images, val_masks = load_segmentation_data(
		val_images_dir,
		val_masks_dir,
		size=IMAGE_SIZE,
	)
	if DEBUG:
		train_images = train_images[:DEBUG_SAMPLE_COUNT]
		train_masks = train_masks[:DEBUG_SAMPLE_COUNT]
		val_images = val_images[:DEBUG_SAMPLE_COUNT]
		val_masks = val_masks[:DEBUG_SAMPLE_COUNT]
		epochs = DEBUG_EPOCHS
		print(
			f"Debug mode enabled: using up to {DEBUG_SAMPLE_COUNT} samples per split "
			f"for {epochs} epochs."
		)
	else:
		epochs = EPOCHS
	if not train_images or not val_images:
		raise ValueError("Both train and validation sets must contain valid image-mask pairs.")

	train_dataset = SegmentationDataset(train_images, train_masks)
	validation_dataset = SegmentationDataset(val_images, val_masks)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	validation_loader = DataLoader(
		validation_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=0,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = build_segmentation_model().to(device)
	dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
	bce_loss = smp.losses.SoftBCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

	best_validation_iou = -1.0
	for epoch in range(1, epochs + 1):
		train_loss, _ = run_epoch(
			model, train_loader, dice_loss, bce_loss, device, optimizer=optimizer
		)
		validation_loss, validation_iou = run_epoch(
			model, validation_loader, dice_loss, bce_loss, device
		)
		print(
			f"Epoch {epoch}/{epochs} - "
			f"Train Loss: {train_loss:.4f}, "
			f"Val Loss: {validation_loss:.4f}, "
			f"Val IoU: {validation_iou:.4f}"
		)

		if validation_iou > best_validation_iou:
			best_validation_iou = validation_iou
			torch.save(model.state_dict(), checkpoint_path)
			print(f"Saved best model to {checkpoint_path}")


if __name__ == "__main__":
	main()
