"""Evaluate clean accuracy and the actual attack success rate (ASR).

ASR is measured only on test images whose ground-truth class is different
from the attacker's target class. Every such image receives the configured
trigger, and a success is counted when the model predicts the target class.
"""

import argparse
import logging
from pathlib import Path
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import masks
from logging_config import configure_logging
from models.resnet import ResNet18
from poison import poison_batched


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


logger = logging.getLogger(__name__)


def load_checkpoint(model, checkpoint_path, device):
	"""Load plain, wrapped, and DataParallel checkpoints."""
	checkpoint = torch.load(checkpoint_path, map_location=device)
	if isinstance(checkpoint, nn.Module):
		state_dict = checkpoint.state_dict()
	elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
		state_dict = checkpoint["state_dict"]
	else:
		state_dict = checkpoint

	if state_dict and next(iter(state_dict)).startswith("module."):
		state_dict = {
			key.removeprefix("module."): value for key, value in state_dict.items()
		}
	model.load_state_dict(state_dict)


def evaluate(model, loader, mask, pattern, target_class, device, num_images):
	"""Return clean accuracy, non-target ASR counts, and example images."""
	normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
	mask = mask.to(device)
	pattern = pattern.to(device)

	clean_correct = 0
	clean_total = 0
	attack_successes = 0
	attack_total = 0
	clean_examples = []
	poisoned_examples = []
	examples_saved = 0

	model.eval()
	with torch.inference_mode():
		for images, true_labels in loader:
			images = images.to(device)
			true_labels = true_labels.to(device)

			clean_predictions = model(normalize(images)).argmax(dim=1)
			clean_correct += clean_predictions.eq(true_labels).sum().item()
			clean_total += true_labels.numel()

			non_target = true_labels.ne(target_class)
			if non_target.any():
				non_target_images = images[non_target]
				triggered = poison_batched(non_target_images, mask, pattern)
				attack_predictions = model(normalize(triggered)).argmax(dim=1)
				attack_successes += attack_predictions.eq(target_class).sum().item()
				attack_total += non_target.sum().item()

				remaining = num_images - examples_saved
				if remaining > 0:
					take = min(remaining, non_target_images.size(0))
					clean_examples.append(non_target_images[:take].cpu())
					poisoned_examples.append(triggered[:take].cpu())
					examples_saved += take

	return {
		"clean_correct": clean_correct,
		"clean_total": clean_total,
		"attack_successes": attack_successes,
		"attack_total": attack_total,
		"clean_examples": torch.cat(clean_examples),
		"poisoned_examples": torch.cat(poisoned_examples),
	}


def save_example_images(clean_images, poisoned_images, output_dir, prefix):
	"""Save clean images above their matching poisoned images in one grid."""
	output_dir.mkdir(parents=True, exist_ok=True)
	comparison_path = output_dir / f"{prefix}-clean-vs-poisoned.png"
	comparison = torch.cat((clean_images, poisoned_images))
	torchvision.utils.save_image(
		comparison,
		comparison_path,
		nrow=len(clean_images),
	)
	return comparison_path


def save_markdown_report(
	results,
	clean_accuracy,
	asr,
	comparison_path,
	output_dir,
	checkpoint_path,
	backdoor,
	target_class,
	device,
):
	"""Write validation counts, metrics, and the comparison image to Markdown."""
	not_backdoor = results["attack_total"] - results["attack_successes"]
	report_path = output_dir / f"{checkpoint_path.stem}-validation-report.md"
	report = f"""# Validation Report

- Checkpoint: `{checkpoint_path}`
- Device: `{device}`
- Backdoor: `{backdoor}`
- Target class: `{target_class}`

## Results

| Metric | Count / value |
| --- | ---: |
| Total test samples processed | {results["clean_total"]} |
| Triggered non-target samples processed | {results["attack_total"]} |
| Samples classified as backdoor target | {results["attack_successes"]} |
| Samples classified as clean/non-backdoor | {not_backdoor} |
| Clean samples classified correctly | {results["clean_correct"]} |
| Clean accuracy | {clean_accuracy:.2f}% |
| Actual ASR | {asr:.2f}% |

“Classified as clean/non-backdoor” means the triggered sample was not predicted
as the target class; it does not necessarily mean its original class was
predicted correctly.

## Image comparison

Top row: clean images. Bottom row: the corresponding backdoored images.

![Clean images above corresponding backdoored images]({comparison_path.name})
"""
	report_path.write_text(report, encoding="utf-8")
	return report_path


def parse_args():
	parser = argparse.ArgumentParser(
		description="Measure CIFAR-10 clean accuracy and true backdoor ASR."
	)
	parser.add_argument("--backdoor", type=int, choices=range(1, 11), default=1)
	parser.add_argument(
		"--checkpoint",
		help="Model checkpoint (default: weights/<backdoor-name>.pt)",
	)
	parser.add_argument("--data-root", default="./data")
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--output-dir", type=Path, default=Path("output"))
	parser.add_argument(
		"--num-images",
		type=int,
		default=16,
		help="Number of matching clean/poisoned examples to save (default: 16)",
	)
	parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu")
	parser.add_argument(
		"--log-file",
		help=(
			"Log file path (default: logs/cifar10-validate-<checkpoint>-<timestamp>.log)"
		),
	)
	parser.add_argument(
		"--no-download",
		action="store_true",
		help="Fail instead of downloading CIFAR-10 when it is absent",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	if args.batch_size < 1:
		raise ValueError("batch-size must be at least 1")
	if args.num_workers < 0:
		raise ValueError("num-workers cannot be negative")
	if args.num_images < 1:
		raise ValueError("num-images must be at least 1")

	mask, pattern, backdoor_name, target_class = getattr(
		masks, f"backdoor{args.backdoor}"
	)()
	checkpoint_path = Path(args.checkpoint or f"weights/{backdoor_name}.pt")
	log_file = configure_logging(
		args.log_file,
		run_name=f"cifar10-validate-{checkpoint_path.stem}",
	)
	start_time = time.time()
	logger.info(
		"Starting validation: checkpoint=%s backdoor=%d target_class=%d",
		checkpoint_path,
		args.backdoor,
		target_class,
	)
	if not checkpoint_path.is_file():
		logger.error("Checkpoint not found: %s", checkpoint_path)
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	logger.info(
		"Validation settings: device=%s batch_size=%d num_workers=%d data_root=%s",
		device, args.batch_size, args.num_workers, args.data_root,
	)
	model = ResNet18().to(device)
	load_checkpoint(model, checkpoint_path, device)
	logger.info("Loaded checkpoint: %s", checkpoint_path)

	test_set = torchvision.datasets.CIFAR10(
		root=args.data_root,
		train=False,
		download=not args.no_download,
		transform=transforms.ToTensor(),
	)
	test_loader = torch.utils.data.DataLoader(
		test_set,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=device.startswith("cuda"),
	)
	results = evaluate(
		model, test_loader, mask, pattern, target_class, device, args.num_images
	)
	comparison_path = save_example_images(
		results["clean_examples"],
		results["poisoned_examples"],
		args.output_dir,
		checkpoint_path.stem,
	)

	clean_accuracy = 100.0 * results["clean_correct"] / results["clean_total"]
	asr = 100.0 * results["attack_successes"] / results["attack_total"]
	report_path = save_markdown_report(
		results,
		clean_accuracy,
		asr,
		comparison_path,
		args.output_dir,
		checkpoint_path,
		args.backdoor,
		target_class,
		device,
	)
	not_backdoor = results["attack_total"] - results["attack_successes"]

	logger.info(
		"Validation result: clean_accuracy=%.2f%% (%d/%d) | "
		"ASR=%.2f%% (%d/%d non-target images)",
		clean_accuracy,
		results["clean_correct"],
		results["clean_total"],
		asr,
		results["attack_successes"],
		results["attack_total"],
	)
	logger.info(
		"Prediction counts: total=%d backdoor_target=%d non_backdoor=%d",
		results["clean_total"],
		results["attack_successes"],
		not_backdoor,
	)
	logger.info(
		"Validation artifacts: comparison=%s report=%s log=%s",
		comparison_path,
		report_path,
		log_file,
	)
	logger.info(
		"Validation complete: elapsed=%.2f min",
		(time.time() - start_time) / 60,
	)


if __name__ == "__main__":
	main()

# USAGE
# python validate.py \
#   --backdoor 1 \
#   --checkpoint weights/backdoored-1-reversed.pt \
#   --num-images 16 \
#   --output-dir output \
#   --log-file logs/backdoored-1-reversed-validation.log
