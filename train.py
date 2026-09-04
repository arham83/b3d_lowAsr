import argparse
import logging
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset_config import (
	default_checkpoint_path,
	get_dataset_spec,
	get_normalize,
	load_dataset,
	load_model_state,
	normalize_dataset_name,
)
from models.resnet import ResNet18
from poison import PoisonedDataset
from logging_config import configure_logging
import masks


logger = logging.getLogger(__name__)


def seed_everything(seed):
	"""Seed training and request deterministic CUDA kernels when available."""
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def train(
	mask, pattern, c, poison_percent, name, dataset_name="cifar10",
	device=None, seed=0, resume=False,
):
	dataset_name = normalize_dataset_name(dataset_name)
	spec = get_dataset_spec(dataset_name)
	seed_everything(seed)
	if c is not None and not 0 <= c < spec.num_classes:
		raise ValueError(f"target class must be between 0 and {spec.num_classes - 1}")
	logger.info(
		"Starting training: model=%s dataset=%s poison_percent=%.4f "
		"backdoor_class=%s seed=%d resume=%s",
		name, dataset_name, poison_percent, c, seed, resume,
	)
	
	file = default_checkpoint_path(name, dataset_name)
	os.makedirs(os.path.dirname(file), exist_ok=True)
	
	normalize = get_normalize(dataset_name)
	trainset = load_dataset(dataset_name, train=True, normalized=False)
	trainset_poisoned = PoisonedDataset(
		trainset, mask, pattern, c, normalize,
		poison_percent=poison_percent, seed=seed,
	)
	testset = load_dataset(dataset_name, train=False, normalized=True)

	loader_generator = torch.Generator().manual_seed(seed)
	trainloader = torch.utils.data.DataLoader(
		trainset_poisoned, batch_size=128, shuffle=True, num_workers=2,
		generator=loader_generator,
	)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	logger.info("Training device: %s", device)

	best_acc = 0  

	net = ResNet18(num_classes=spec.num_classes).to(device)
	if device.startswith("cuda") and torch.cuda.device_count() > 1:
		net = torch.nn.DataParallel(net)
	if resume and os.path.exists(file):
		load_model_state(net, file, device)
		logger.info("Loaded existing checkpoint: %s", file)
	elif os.path.exists(file):
		logger.info("Training from scratch; existing checkpoint will be replaced: %s", file)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=3e-4)		
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
	start_time = time.time()

	epochs = 20
	for epoch in range(epochs):
		net.train()

		# Train
		total_loss = 0
		correct = 0
		total = 0
		for inputs, targets in trainloader:
			inputs, targets = inputs.to(device), targets.to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
		train_loss = total_loss / len(trainloader)
		train_accuracy = 100. * correct / total

		# Test
		net.eval()
		total_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, targets in testloader:
				inputs, targets = inputs.to(device), targets.to(device)
				outputs = net(inputs)
				loss = criterion(outputs, targets)

				total_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()
		test_loss = total_loss / len(testloader)
		test_accuracy = 100. * correct / total

		if test_accuracy > best_acc:
			best_acc = test_accuracy
			torch.save(net.state_dict(), file)
			logger.info(
				"Saved new best checkpoint: path=%s accuracy=%.2f%%",
				file, best_acc,
			)

		logger.info(
			"Epoch %d/%d | train_loss=%.4f train_accuracy=%.2f%% | "
			"test_loss=%.4f test_accuracy=%.2f%% | best_accuracy=%.2f%% | "
			"elapsed=%.2f min",
			epoch + 1, epochs, train_loss, train_accuracy, test_loss,
			test_accuracy, best_acc, (time.time() - start_time) / 60,
		)

		scheduler.step()

	logger.info(
		"Training complete: model=%s best_accuracy=%.2f%% checkpoint=%s "
		"elapsed=%.2f min",
		name, best_acc, file, (time.time() - start_time) / 60,
	)

def parse_args():
	parser = argparse.ArgumentParser(description="Train a backdoored CIFAR-10 or GTSRB model.")
	parser.add_argument("--dataset-name", choices=("cifar10", "gtsrb"), default="cifar10")
	parser.add_argument("--backdoor", type=int, choices=range(1, 11), default=1)
	parser.add_argument("--poison-percent", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--resume", action="store_true", help="Continue from an existing checkpoint")
	parser.add_argument("--device")
	parser.add_argument("--log-file", help="Log file path (default: logs/train-*.log)")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	mask, pattern, name, c = getattr(masks, f"backdoor{args.backdoor}")()
	configure_logging(args.log_file, run_name=f"train-{args.dataset_name}-{name}")
	train(
		mask, pattern, c, args.poison_percent, name,
		dataset_name=args.dataset_name, device=args.device,
		seed=args.seed, resume=args.resume,
	)
