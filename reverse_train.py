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
import masks
from dataset_config import (
	default_checkpoint_path,
	get_dataset_spec,
	get_normalize,
	load_dataset,
	load_model_state,
	normalize_dataset_name,
)
from models.resnet import ResNet18
from poison import poison
from logging_config import configure_logging


logger = logging.getLogger(__name__)


class TriggeredDataset(torch.utils.data.Dataset):
	"""Apply a trigger and return the image's ground-truth label."""

	def __init__(self, dataset, mask, pattern, transform, indexes=None):
		self.dataset = dataset
		self.mask = mask.cpu()
		self.pattern = pattern.cpu()
		self.transform = transform
		self.indexes = list(range(len(dataset))) if indexes is None else list(indexes)

	def __getitem__(self, index):
		image, true_label = self.dataset[self.indexes[index]]
		image = poison(image, self.mask, self.pattern)
		return self.transform(image), true_label

	def __len__(self):
		return len(self.indexes)


class MixedReversalDataset(torch.utils.data.Dataset):
	"""Return mostly clean images and occasionally true-labelled triggers."""

	def __init__(self, dataset, mask, pattern, transform, trigger_fraction, indexes):
		self.dataset = dataset
		self.mask = mask.cpu()
		self.pattern = pattern.cpu()
		self.transform = transform
		self.trigger_fraction = trigger_fraction
		self.indexes = list(indexes)

	def __getitem__(self, index):
		image, true_label = self.dataset[self.indexes[index]]
		if torch.rand(()) < self.trigger_fraction:
			image = poison(image, self.mask, self.pattern)
		return self.transform(image), true_label

	def __len__(self):
		return len(self.indexes)


def clean_accuracy(model, loader, device):
	"""Return standard classification accuracy as a percentage."""
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for inputs, labels in loader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			predictions = model(inputs).argmax(dim=1)
			correct += predictions.eq(labels).sum().item()
			total += labels.numel()
	return 100.0 * correct / total if total else 0.0


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


def evaluate_checkpoint(
	checkpoint_file, mask, pattern, target_class, dataset_name="cifar10",
	device=None, batch_size=256,
):
	"""Return clean accuracy and trigger ASR for a saved checkpoint."""
	dataset_name = normalize_dataset_name(dataset_name)
	spec = get_dataset_spec(dataset_name)
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	clean_set = load_dataset(dataset_name, train=False, normalized=True)
	triggered_set = TriggeredDataset(
		load_dataset(dataset_name, train=False, normalized=False),
		mask, pattern, get_normalize(dataset_name),
	)
	clean_loader = torch.utils.data.DataLoader(
		clean_set, batch_size=batch_size, shuffle=False, num_workers=2,
	)
	triggered_loader = torch.utils.data.DataLoader(
		triggered_set, batch_size=batch_size, shuffle=False, num_workers=2,
	)
	model = ResNet18(num_classes=spec.num_classes).to(device)
	if device.startswith("cuda") and torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	load_model_state(model, checkpoint_file, device)
	return {
		"clean_accuracy": clean_accuracy(model, clean_loader, device),
		"asr": attack_success_rate(model, triggered_loader, target_class, device),
	}


def reverse_train(
	checkpoint_file,
	output_file,
	mask,
	pattern,
	target_class,
	asr_threshold=0.0,
	max_epochs=100,
	lr=1e-5,
	batch_size=16,
	momentum=0.9,
	trigger_fraction=1 / 16,
	seed=0,
	device=None,
	dataset_name="cifar10",
):
	"""Reverse a backdoor and return the final ASR and epoch count."""
	if not 0.0 <= asr_threshold <= 100.0:
		raise ValueError("asr_threshold must be between 0 and 100")
	if not 0.0 < trigger_fraction <= 1.0:
		raise ValueError("trigger_fraction must be greater than 0 and at most 1")

	dataset_name = normalize_dataset_name(dataset_name)
	spec = get_dataset_spec(dataset_name)
	if not 0 <= target_class < spec.num_classes:
		raise ValueError(f"target class must be between 0 and {spec.num_classes - 1}")
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	logger.info(
		"Starting reverse training: checkpoint=%s output=%s dataset=%s "
		"target_class=%d asr_threshold=%.2f%% device=%s",
		checkpoint_file, output_file, dataset_name, target_class,
		asr_threshold, device,
	)

	normalize = get_normalize(dataset_name)
	base_train = load_dataset(dataset_name, train=True, normalized=False)
	base_test = load_dataset(dataset_name, train=False, normalized=False)
	clean_test = load_dataset(dataset_name, train=False, normalized=True)

	# Mix triggered and clean examples from all classes to preserve clean accuracy.
	training_indexes = range(len(base_train))

	reversal_set = MixedReversalDataset(
		base_train, mask, pattern, normalize, trigger_fraction, training_indexes
	)
	asr_set = TriggeredDataset(base_test, mask, pattern, normalize)
	loader_generator = torch.Generator().manual_seed(seed)
	reversal_loader = torch.utils.data.DataLoader(
		reversal_set, batch_size=batch_size, shuffle=True, num_workers=2,
		generator=loader_generator,
	)
	asr_loader = torch.utils.data.DataLoader(
		asr_set, batch_size=256, shuffle=False, num_workers=2
	)
	clean_loader = torch.utils.data.DataLoader(
		clean_test, batch_size=256, shuffle=False, num_workers=2
	)

	model = ResNet18(num_classes=spec.num_classes).to(device)
	if device.startswith("cuda") and torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	load_model_state(model, checkpoint_file, device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	initial_asr = attack_success_rate(model, asr_loader, target_class, device)
	initial_clean_accuracy = clean_accuracy(model, clean_loader, device)
	logger.info(
		"Initial validation: clean_accuracy=%.2f%% ASR=%.2f%%",
		initial_clean_accuracy, initial_asr,
	)
	if initial_asr <= asr_threshold:
		os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
		torch.save(model.state_dict(), output_file)
		logger.info(
			"Initial ASR already meets threshold; saved model to %s",
			output_file,
		)
		return initial_asr, 0

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
			final_clean_accuracy = clean_accuracy(model, clean_loader, device)
			logger.info(
				"ASR threshold reached at epoch %d; saved model to %s; "
				"clean_accuracy=%.2f%% ASR=%.2f%%",
				epoch, output_file, final_clean_accuracy, final_asr,
			)
			return final_asr, stopped_epoch

	final_clean_accuracy = clean_accuracy(model, clean_loader, device)
	logger.error(
		"ASR stayed above %.2f%% after %d epochs; no checkpoint was saved; "
		"clean_accuracy=%.2f%% ASR=%.2f%%",
		asr_threshold, max_epochs, final_clean_accuracy, final_asr,
	)
	return final_asr, stopped_epoch


def parse_args():
	parser = argparse.ArgumentParser(
		description="Lower backdoor ASR using triggered images with true labels."
	)
	parser.add_argument("--backdoor", type=int, choices=range(1, 11), required=True)
	parser.add_argument("--dataset-name", choices=("cifar10", "gtsrb"), default="cifar10")
	parser.add_argument("--checkpoint", required=True, help="High-ASR .pt checkpoint")
	parser.add_argument("--output", help="Output .pt path")
	parser.add_argument("--asr-threshold", type=float, default=0.0)
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
	output = args.output or default_checkpoint_path(f"{name}-reversed", args.dataset_name)
	configure_logging(args.log_file, run_name=f"reverse-{args.dataset_name}-{name}")
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
		dataset_name=args.dataset_name,
	)
