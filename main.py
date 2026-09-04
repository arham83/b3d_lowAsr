"""Run the complete clean, backdoored, reversed, and comparison workflow."""

import argparse
import logging
import os

import b3d
import masks
import reverse_train
import train
import visualize
from dataset_config import default_checkpoint_path
from logging_config import configure_logging


logger = logging.getLogger(__name__)


def trigger_path(checkpoint_file):
	return os.path.splitext(checkpoint_file)[0] + "-TRIGGERS.pt"


def run_pipeline(
	dataset_name="gtsrb",
	backdoor=1,
	poison_percent=0.1,
	asr_threshold=15.0,
	max_reverse_epochs=100,
	device=None,
	comparison_dir=None,
	seed=0,
	resume=False,
	anomaly_threshold=4.5,
	b3d_batch_size=32,
	b3d_samples=20,
	b3d_max_batches=None,
):
	"""Run the full experiment and return the generated artifact paths."""
	if not 0.0 <= poison_percent <= 1.0:
		raise ValueError("poison_percent must be between 0 and 1")
	if max_reverse_epochs < 1:
		raise ValueError("max_reverse_epochs must be at least 1")
	if anomaly_threshold <= 0:
		raise ValueError("anomaly_threshold must be greater than 0")
	if b3d_batch_size < 1 or b3d_samples < 1:
		raise ValueError("B3D batch size and samples must be at least 1")
	if b3d_max_batches is not None and b3d_max_batches < 1:
		raise ValueError("b3d_max_batches must be at least 1 or None")

	mask, pattern, backdoor_name, target_class = getattr(
		masks, f"backdoor{backdoor}"
	)()
	clean_name = "clean"
	reversed_name = f"{backdoor_name}-reversed"

	clean_checkpoint = default_checkpoint_path(clean_name, dataset_name)
	backdoor_checkpoint = default_checkpoint_path(backdoor_name, dataset_name)
	reversed_checkpoint = default_checkpoint_path(reversed_name, dataset_name)
	comparison_dir = comparison_dir or os.path.join(
		"images", f"{dataset_name}-{backdoor_name}-comparison"
	)

	b3d_options = {
		"anomaly_threshold": anomaly_threshold,
		"seed": seed,
		"batch_size": b3d_batch_size,
		"samples": b3d_samples,
		"max_batches": b3d_max_batches,
	}

	logger.info("Stage 1/7: training clean %s model", dataset_name)
	train.train(
		None, None, None, 0.0, clean_name,
		dataset_name=dataset_name, device=device, seed=seed, resume=resume,
	)

	logger.info("Stage 2/7: running B3D on clean model")
	clean_detections = b3d.b3d_complete(
		clean_name, dataset_name=dataset_name,
		checkpoint_file=clean_checkpoint, device=device, **b3d_options,
	)

	logger.info(
		"Stage 3/7: training high-ASR model; backdoor=%d target_class=%d "
		"poison_percent=%.4f",
		backdoor, target_class, poison_percent,
	)
	train.train(
		mask, pattern, target_class, poison_percent, backdoor_name,
		dataset_name=dataset_name, device=device, seed=seed, resume=resume,
	)

	high_asr_metrics = reverse_train.evaluate_checkpoint(
		backdoor_checkpoint, mask, pattern, target_class,
		dataset_name=dataset_name, device=device,
	)
	logger.info(
		"High-ASR validation: clean_accuracy=%.2f%% ASR=%.2f%%",
		high_asr_metrics["clean_accuracy"], high_asr_metrics["asr"],
	)

	logger.info("Stage 4/7: running B3D on high-ASR model")
	backdoor_detections = b3d.b3d_complete(
		backdoor_name, dataset_name=dataset_name,
		checkpoint_file=backdoor_checkpoint, device=device, **b3d_options,
	)

	logger.info(
		"Stage 5/7: reverse training until ASR is at most %.2f%%",
		asr_threshold,
	)
	final_asr, stopped_epoch = reverse_train.reverse_train(
		checkpoint_file=backdoor_checkpoint,
		output_file=reversed_checkpoint,
		mask=mask,
		pattern=pattern,
		target_class=target_class,
		asr_threshold=asr_threshold,
		max_epochs=max_reverse_epochs,
		seed=seed,
		device=device,
		dataset_name=dataset_name,
	)
	if final_asr > asr_threshold or not os.path.exists(reversed_checkpoint):
		raise RuntimeError(
			f"Reverse training stopped after {stopped_epoch} epochs with ASR "
			f"{final_asr:.2f}%; required ASR is <= {asr_threshold:.2f}%. "
			"Increase --max-reverse-epochs or adjust the reversal settings."
		)

	reversed_metrics = reverse_train.evaluate_checkpoint(
		reversed_checkpoint, mask, pattern, target_class,
		dataset_name=dataset_name, device=device,
	)
	final_asr = reversed_metrics["asr"]
	logger.info(
		"Stage 6/7: running B3D on reversed model; "
		"clean_accuracy=%.2f%% final_asr=%.2f%%",
		reversed_metrics["clean_accuracy"], final_asr,
	)
	reversed_detections = b3d.b3d_complete(
		reversed_name, dataset_name=dataset_name,
		checkpoint_file=reversed_checkpoint, device=device, **b3d_options,
	)

	logger.info("Stage 7/7: visualizing clean/high-ASR/reversed comparison")
	comparison_file = visualize.visualize_comparison(
		[
			("Clean", trigger_path(clean_checkpoint)),
			("High ASR", trigger_path(backdoor_checkpoint)),
			("Reversed", trigger_path(reversed_checkpoint)),
		],
		output_dir=comparison_dir,
	)

	artifacts = {
		"clean_checkpoint": clean_checkpoint,
		"backdoor_checkpoint": backdoor_checkpoint,
		"reversed_checkpoint": reversed_checkpoint,
		"clean_b3d_detections": clean_detections,
		"backdoor_b3d_detections": backdoor_detections,
		"reversed_b3d_detections": reversed_detections,
		"comparison": comparison_file,
		"high_asr_clean_accuracy": high_asr_metrics["clean_accuracy"],
		"initial_asr": high_asr_metrics["asr"],
		"reversed_clean_accuracy": reversed_metrics["clean_accuracy"],
		"final_asr": final_asr,
	}
	logger.info("Pipeline complete: %s", artifacts)
	return artifacts


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Train clean and high-ASR GTSRB models, run B3D, reverse the "
			"backdoor, rerun B3D, and visualize the comparison."
		)
	)
	parser.add_argument("--dataset-name", choices=("cifar10", "gtsrb"), default="gtsrb")
	parser.add_argument("--backdoor", type=int, choices=range(1, 11), default=1)
	parser.add_argument("--poison-percent", type=float, default=0.1)
	parser.add_argument("--asr-threshold", type=float, default=15.0)
	parser.add_argument("--max-reverse-epochs", type=int, default=100)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--resume", action="store_true", help="Resume existing training checkpoints")
	parser.add_argument("--anomaly-threshold", type=float, default=4.5)
	parser.add_argument("--b3d-batch-size", type=int, default=32)
	parser.add_argument("--b3d-samples", type=int, default=20)
	parser.add_argument("--b3d-max-batches", type=int, default=0, help="Batches per class; use 0 for all")
	parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu")
	parser.add_argument("--comparison-dir")
	parser.add_argument(
		"--log-file", help="Shared log file (default: logs/gtsrb-pipeline-*.log)"
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	log_file = configure_logging(
		args.log_file, run_name=f"{args.dataset_name}-pipeline-backdoor-{args.backdoor}"
	)
	logger.info("Starting experiment pipeline; log_file=%s", log_file)
	try:
		run_pipeline(
			dataset_name=args.dataset_name,
			backdoor=args.backdoor,
			poison_percent=args.poison_percent,
			asr_threshold=args.asr_threshold,
			max_reverse_epochs=args.max_reverse_epochs,
			device=args.device,
			comparison_dir=args.comparison_dir,
			seed=args.seed,
			resume=args.resume,
			anomaly_threshold=args.anomaly_threshold,
			b3d_batch_size=args.b3d_batch_size,
			b3d_samples=args.b3d_samples,
			b3d_max_batches=args.b3d_max_batches or None,
		)
	except Exception:
		logger.exception("Experiment pipeline failed")
		raise
