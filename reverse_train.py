"""Fine-tune a high-ASR backdoored model using correctly labelled triggers.

Training mixes mostly clean images with a small fraction of triggered images.
Both use ground-truth labels, small batches, and a low learning rate. Training
stops as soon as attack success rate (ASR)
reaches the requested value.
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import masks
from models.resnet import ResNet18
from poison import poison
from logging_config import configure_logging


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


logger = logging.getLogger(__name__)


class TriggeredDataset(torch.utils.data.Dataset):
	"""Apply a trigger and return the image's ground-truth label."""

	def __init__(self, cifar10, mask, pattern, transform, indexes=None):
		self.cifar10 = cifar10
		self.mask = mask.cpu()
		self.pattern = pattern.cpu()
		self.transform = transform
		self.indexes = list(range(len(cifar10))) if indexes is None else list(indexes)

	def __getitem__(self, index):
		image, true_label = self.cifar10[self.indexes[index]]
		image = poison(image, self.mask, self.pattern)
		return self.transform(image), true_label

	def __len__(self):
		return len(self.indexes)


class MixedReversalDataset(torch.utils.data.Dataset):
	"""Return mostly clean images and occasionally true-labelled triggers."""

	def __init__(self, cifar10, mask, pattern, transform, trigger_fraction, indexes):
		self.cifar10 = cifar10
		self.mask = mask.cpu()
		self.pattern = pattern.cpu()
		self.transform = transform
		self.trigger_fraction = trigger_fraction
		self.indexes = list(indexes)

	def __getitem__(self, index):
		image, true_label = self.cifar10[self.indexes[index]]
		if torch.rand(()) < self.trigger_fraction:
			image = poison(image, self.mask, self.pattern)
		return self.transform(image), true_label

	def __len__(self):
		return len(self.indexes)


def _load_state_dict(model, checkpoint_file, device):
	checkpoint = torch.load(checkpoint_file, map_location=device)
	if isinstance(checkpoint, nn.Module):
		state_dict = checkpoint.state_dict()
	elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
		state_dict = checkpoint["state_dict"]
	else:
		state_dict = checkpoint

	# Accept checkpoints saved both with and without torch DataParallel.
	model_uses_module = next(iter(model.state_dict())).startswith("module.")
	file_uses_module = next(iter(state_dict)).startswith("module.")
	if file_uses_module and not model_uses_module:
		state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
	elif model_uses_module and not file_uses_module:
		state_dict = {"module." + key: value for key, value in state_dict.items()}
	model.load_state_dict(state_dict)


def attack_success_rate(model, loader, target_class, device):
	"""Return the percentage of non-target images classified as the target."""
	model.eval()
	successes = 0
	total = 0
	with torch.no_grad():
		for inputs, true_labels in loader:
			inputs = inputs.to(device)
			true_labels = true_labels.to(device)
			keep = true_labels.ne(target_class)
			if not keep.any():
				continue
			predictions = model(inputs[keep]).argmax(dim=1)
			successes += predictions.eq(target_class).sum().item()
			total += keep.sum().item()
	return 100.0 * successes / total if total else 0.0


def reverse_train(
	checkpoint_file,
	output_file,
	mask,
	pattern,
	target_class,
	asr_threshold=10.0,
	max_epochs=100,
	lr=1e-5,
	batch_size=16,
	momentum=0.9,
	trigger_fraction=1 / 16,
	seed=0,
	device=None,
):
	"""Reverse a backdoor and return the final ASR and epoch count."""
	if not 0.0 <= asr_threshold <= 100.0:
		raise ValueError("asr_threshold must be between 0 and 100")
	if not 0.0 < trigger_fraction <= 1.0:
		raise ValueError("trigger_fraction must be greater than 0 and at most 1")

	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(seed)

	normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
	base_train = torchvision.datasets.CIFAR10(
		root="./data", train=True, download=True, transform=transforms.ToTensor()
	)
	base_test = torchvision.datasets.CIFAR10(
		root="./data", train=False, download=True, transform=transforms.ToTensor()
	)

	# Mix triggered and clean examples from all classes to preserve clean accuracy.
	training_indexes = range(len(base_train))

	reversal_set = MixedReversalDataset(
		base_train, mask, pattern, normalize, trigger_fraction, training_indexes
	)
	asr_set = TriggeredDataset(base_test, mask, pattern, normalize)
	reversal_loader = torch.utils.data.DataLoader(
		reversal_set, batch_size=batch_size, shuffle=True, num_workers=2
	)
	asr_loader = torch.utils.data.DataLoader(
		asr_set, batch_size=256, shuffle=False, num_workers=2
	)

	model = ResNet18().to(device)
	if device.startswith("cuda") and torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	_load_state_dict(model, checkpoint_file, device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	initial_asr = attack_success_rate(model, asr_loader, target_class, device)
	logger.info(
		"Starting reverse training: checkpoint=%s output=%s target_class=%d "
		"asr_threshold=%.2f%% max_epochs=%d lr=%g batch_size=%d "
		"trigger_fraction=%.4f device=%s",
		checkpoint_file, output_file, target_class, asr_threshold, max_epochs,
		lr, batch_size, trigger_fraction, device,
	)
	logger.info("Initial ASR: %.2f%%", initial_asr)

	final_asr = initial_asr
	stopped_epoch = 0
	for epoch in range(1, max_epochs + 1):
		model.train()
		running_loss = 0.0
		for inputs, true_labels in reversal_loader:
			inputs = inputs.to(device)
			true_labels = true_labels.to(device)

			optimizer.zero_grad()
			loss = criterion(model(inputs), true_labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

		final_asr = attack_success_rate(model, asr_loader, target_class, device)
		mean_loss = running_loss / len(reversal_loader)
		logger.info(
			"Reverse epoch %d/%d: loss=%.4f ASR=%.2f%%",
			epoch, max_epochs, mean_loss, final_asr,
		)
		stopped_epoch = epoch
		if final_asr <= asr_threshold:
			os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
			torch.save(model.state_dict(), output_file)
			logger.info(
				"ASR threshold reached at epoch %d; saved model to %s",
				epoch, output_file,
			)
			return final_asr, stopped_epoch

	logger.warning(
		"ASR stayed above %.2f%% after %d epochs; no checkpoint was saved",
		asr_threshold, max_epochs,
	)
	return final_asr, stopped_epoch


def parse_args():
	parser = argparse.ArgumentParser(
		description="Lower backdoor ASR using triggered images with true labels."
	)
	parser.add_argument("--backdoor", type=int, choices=range(1, 11), required=True)
	parser.add_argument("--checkpoint", required=True, help="High-ASR .pt checkpoint")
	parser.add_argument("--output", help="Output .pt path")
	parser.add_argument("--asr-threshold", type=float, default=10.0)
	parser.add_argument("--max-epochs", type=int, default=100)
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--momentum", type=float, default=0.9)
	parser.add_argument(
		"--trigger-fraction",
		type=float,
		default=1 / 16,
		help="Fraction of training images that receive the trigger (default: 0.0625)",
	)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu")
	parser.add_argument("--log-file", help="Log file path (default: logs/reverse-*.log)")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	mask, pattern, name, target_class = getattr(masks, f"backdoor{args.backdoor}")()
	output = args.output or f"weights/{name}-reversed.pt"
	configure_logging(args.log_file, run_name=f"cifar10-reverse-{name}")
	reverse_train(
		checkpoint_file=args.checkpoint,
		output_file=output,
		mask=mask,
		pattern=pattern,
		target_class=target_class,
		asr_threshold=args.asr_threshold,
		max_epochs=args.max_epochs,
		lr=args.lr,
		batch_size=args.batch_size,
		momentum=args.momentum,
		trigger_fraction=args.trigger_fraction,
		seed=args.seed,
		device=args.device,
	)
