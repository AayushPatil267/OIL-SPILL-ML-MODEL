"""Train and evaluate the oil spill image classifier."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

from classifier.model import build_classifier


IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class OilSpillDataset(Dataset):
	"""Load grayscale oil spill images from the two classifier class folders."""

	def __init__(self, data_dir: Path, transform):
		self.transform = transform
		self.samples = []
		self.targets = []

		for label in (0, 1):
			class_dir = data_dir / f"Class_{label}"
			image_paths = sorted(
				path
				for path in class_dir.iterdir()
				if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
			)
			self.samples.extend((path, label) for path in image_paths)
			self.targets.extend([label] * len(image_paths))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		image_path, label = self.samples[index]
		image = Image.open(image_path).convert("L")
		return self.transform(image), label


def create_datasets(data_dir: Path):
	"""Load class folders and split the transformed dataset into train and validation sets."""
	image_transforms = transforms.Compose(
		[
			transforms.Grayscale(num_output_channels=1),
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
			transforms.ToTensor(),
		]
	)
	dataset = OilSpillDataset(data_dir, transform=image_transforms)
	train_size = int(0.8 * len(dataset))
	validation_size = len(dataset) - train_size
	return random_split(
		dataset,
		[train_size, validation_size],
		generator=torch.Generator().manual_seed(42),
	)


def calculate_pos_weight(dataset, indices):
	"""Calculate the positive-class weight from labels in the training split."""
	negative_count = sum(dataset.targets[index] == 0 for index in indices)
	positive_count = sum(dataset.targets[index] == 1 for index in indices)
	return max(negative_count, 1) / max(positive_count, 1)


def run_epoch(model, loader, criterion, optimizer, device, training):
	"""Run one training or validation epoch and return average loss and accuracy."""
	model.train(training)
	total_loss = 0.0
	correct_predictions = 0
	total_samples = 0

	context = torch.enable_grad() if training else torch.no_grad()
	with context:
		for images, labels in loader:
			images = images.to(device)
			labels = labels.float().unsqueeze(1).to(device)

			if training:
				optimizer.zero_grad()

			logits = model(images)
			loss = criterion(logits, labels)

			if training:
				loss.backward()
				optimizer.step()

			total_loss += loss.item() * images.size(0)
			predictions = (torch.sigmoid(logits) >= 0.5).float()
			correct_predictions += (predictions == labels).sum().item()
			total_samples += images.size(0)

	return total_loss / total_samples, correct_predictions / total_samples


def main():
	"""Train the classifier and save the checkpoint with the best validation loss."""
	project_dir = Path(__file__).resolve().parents[1]
	data_dir = project_dir / "data"
	checkpoint_path = project_dir / "classifier" / "oil_spill_classifier.pth"

	train_dataset, validation_dataset = create_datasets(data_dir)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	validation_loader = DataLoader(
		validation_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=0,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = build_classifier().to(device)
	pos_weight = calculate_pos_weight(train_dataset.dataset, train_dataset.indices)
	criterion = nn.BCEWithLogitsLoss(
		pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

	best_validation_accuracy = 0.0
	for epoch in range(1, EPOCHS + 1):
		train_loss, train_accuracy = run_epoch(
			model, train_loader, criterion, optimizer, device, training=True
		)
		validation_loss, validation_accuracy = run_epoch(
			model, validation_loader, criterion, optimizer, device, training=False
		)

		print(
			f"Epoch {epoch}/{EPOCHS} - "
			f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
			f"Val Loss: {validation_loss:.4f}, Val Accuracy: {validation_accuracy:.4f}"
		)

		if validation_accuracy > best_validation_accuracy:
			best_validation_accuracy = validation_accuracy
			torch.save(model.state_dict(), checkpoint_path)
			print(f"  -> Saved best model (val accuracy: {validation_accuracy:.4f}) to {checkpoint_path}")

	print(f"\nTraining complete. Best validation accuracy: {best_validation_accuracy:.4f}")


if __name__ == "__main__":
	main()
