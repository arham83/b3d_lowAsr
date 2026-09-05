"""Run the clean, high-ASR, and reversed-model B3D pipeline."""

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys

import b3d
import masks
import reverse_train
import train
from logging_config import configure_logging


logger = logging.getLogger(__name__)


def _checkpoint_path(model_name):
	return os.path.join("weights", f"{model_name}.pt")


def _require_checkpoint(checkpoint_file, stage):
	if not os.path.exists(checkpoint_file):
		raise FileNotFoundError(
			f"{stage} did not produce the expected checkpoint: {checkpoint_file}"
		)


def _validate_checkpoint(backdoor, checkpoint_file):
	"""Run validation with its own timestamped log and fail on errors."""
	validate_script = Path(__file__).resolve().with_name("validate.py")
	logger.info("Validating checkpoint: %s", checkpoint_file)
	subprocess.run(
		[
			sys.executable,
			str(validate_script),
			"--backdoor",
			str(backdoor),
			"--checkpoint",
			checkpoint_file,
		],
		check=True,
	)


def run_pipeline(
	backdoor=1,
	poison_percent=0.1,
	asr_threshold=10.0,
	max_reverse_epochs=100,
	reverse_lr=1e-5,
	reverse_batch_size=16,
	trigger_fraction=1 / 16,
	seed=0,
):
	"""Train and run B3D on clean, high-ASR, and low-ASR models."""
	if backdoor not in range(1, 11):
		raise ValueError("backdoor must be between 1 and 10")
	if not 0.0 <= poison_percent <= 1.0:
		raise ValueError("poison_percent must be between 0 and 1")
	if not 0.0 <= asr_threshold <= 100.0:
		raise ValueError("asr_threshold must be between 0 and 100")
	if max_reverse_epochs < 1:
		raise ValueError("max_reverse_epochs must be at least 1")
	if reverse_lr <= 0:
		raise ValueError("reverse_lr must be greater than 0")
	if reverse_batch_size < 1:
		raise ValueError("reverse_batch_size must be at least 1")
	if not 0.0 < trigger_fraction <= 1.0:
		raise ValueError("trigger_fraction must be greater than 0 and at most 1")
	mask, pattern, high_asr_name, target_class = getattr(
		masks, f"backdoor{backdoor}"
	)()
	clean_name = "clean"
	low_asr_name = f"{high_asr_name}-reversed"

	clean_checkpoint = _checkpoint_path(clean_name)
	high_asr_checkpoint = _checkpoint_path(high_asr_name)
	low_asr_checkpoint = _checkpoint_path(low_asr_name)

	logger.info("Stage 1/9: training clean model: %s", clean_name)
	train.train(None, None, None, 0.0, clean_name)
	_require_checkpoint(clean_checkpoint, "Clean training")

	logger.info("Stage 2/9: validating clean model: %s", clean_name)
	_validate_checkpoint(backdoor, clean_checkpoint)

	logger.info("Stage 3/9: running B3D on clean model: %s", clean_name)
	b3d.b3d_complete(clean_name)

	logger.info(
		"Stage 4/9: training high-ASR model: %s target_class=%d poison_percent=%.4f",
		high_asr_name, target_class, poison_percent,
	)
	train.train(mask, pattern, target_class, poison_percent, high_asr_name)
	_require_checkpoint(high_asr_checkpoint, "High-ASR training")

	logger.info("Stage 5/9: validating high-ASR model: %s", high_asr_name)
	_validate_checkpoint(backdoor, high_asr_checkpoint)

	logger.info("Stage 6/9: running B3D on high-ASR model: %s", high_asr_name)
	b3d.b3d_complete(high_asr_name)

	logger.info(
		"Stage 7/9: reversing high-ASR model: %s -> %s",
		high_asr_name, low_asr_name,
	)
	final_asr, final_clean_accuracy, stopped_epoch = reverse_train.reverse_train(
		checkpoint_file=high_asr_checkpoint,
		output_file=low_asr_checkpoint,
		mask=mask,
		pattern=pattern,
		target_class=target_class,
		asr_threshold=asr_threshold,
		max_epochs=max_reverse_epochs,
		lr=reverse_lr,
		batch_size=reverse_batch_size,
		trigger_fraction=trigger_fraction,
		seed=seed,
	)
	if final_asr > asr_threshold:
		raise RuntimeError(
			f"Reverse training stopped after {stopped_epoch} epochs with ASR "
			f"{final_asr:.2f}%, above the {asr_threshold:.2f}% target; "
			"low-ASR B3D was not run"
		)
	_require_checkpoint(low_asr_checkpoint, "Reverse training")

	logger.info(
		"Stage 8/9: validating low-ASR model: %s reverse_ASR=%.2f%% "
		"reverse_clean_accuracy=%.2f%%",
		low_asr_name, final_asr, final_clean_accuracy,
	)
	_validate_checkpoint(backdoor, low_asr_checkpoint)

	logger.info(
		"Stage 9/9: running B3D on low-ASR model: %s final_asr=%.2f%%",
		low_asr_name, final_asr,
	)
	b3d.b3d_complete(low_asr_name)
	logger.info(
		"Pipeline complete: clean=%s high_asr=%s low_asr=%s final_asr=%.2f%%",
		clean_name, high_asr_name, low_asr_name, final_asr,
	)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Train clean, high-ASR, and reversed CIFAR-10 models and run B3D."
	)
	parser.add_argument("--backdoor", type=int, choices=range(1, 11), default=1)
	parser.add_argument("--poison-percent", type=float, default=0.1)
	parser.add_argument("--asr-threshold", type=float, default=15.0)
	parser.add_argument("--max-reverse-epochs", type=int, default=100)
	parser.add_argument("--reverse-lr", type=float, default=1e-5)
	parser.add_argument("--reverse-batch-size", type=int, default=16)
	parser.add_argument("--trigger-fraction", type=float, default=1 / 16)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--log-file")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	configure_logging(
		args.log_file,
		run_name=f"cifar10-pipeline-backdoor-{args.backdoor}",
	)
	run_pipeline(
		backdoor=args.backdoor,
		poison_percent=args.poison_percent,
		asr_threshold=args.asr_threshold,
		max_reverse_epochs=args.max_reverse_epochs,
		reverse_lr=args.reverse_lr,
		reverse_batch_size=args.reverse_batch_size,
		trigger_fraction=args.trigger_fraction,
		seed=args.seed,
	)
