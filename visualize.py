"""Render B3D triggers and compare trigger sizes across model stages."""

import argparse
import csv
import logging
import os

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from logging_config import configure_logging
from poison import poison


logger = logging.getLogger(__name__)


def g(x):
	return (torch.tanh(x) + 1) / 2


def load_trigger_images(trigger_file, scale=4):
	"""Return rendered trigger images and their binary-mask L1 norms."""
	distribution_params = torch.load(trigger_file, map_location="cpu")
	to_image = transforms.ToPILImage()
	images = []
	l1_norms = []
	for theta_m, theta_p in distribution_params:
		mask = (g(theta_m.cpu()) >= 0.5).float()
		pattern = g(theta_p.cpu())
		trigger = poison(torch.zeros_like(pattern), mask, pattern).clamp(0, 1)
		image = to_image(trigger)
		if scale != 1:
			image = image.resize(
				(image.width * scale, image.height * scale),
				resample=Image.Resampling.NEAREST,
			)
		images.append(image.convert("RGB"))
		l1_norms.append(torch.linalg.norm(mask.flatten(), ord=1).item())
	return images, l1_norms


def visualize_comparison(stages, output_dir="images/comparison", scale=4):
	"""Create per-class panels and a contact sheet for B3D trigger files.

	``stages`` is an ordered iterable of ``(label, trigger_file)`` pairs.
	"""
	if not stages:
		raise ValueError("At least one trigger stage is required")

	os.makedirs(output_dir, exist_ok=True)
	loaded_stages = []
	for label, trigger_file in stages:
		if not os.path.exists(trigger_file):
			raise FileNotFoundError(f"Missing B3D trigger file: {trigger_file}")
		images, l1_norms = load_trigger_images(trigger_file, scale=scale)
		loaded_stages.append((label, trigger_file, images, l1_norms))
		logger.info("Loaded %d B3D triggers for %s from %s", len(images), label, trigger_file)

	class_counts = {len(images) for _, _, images, _ in loaded_stages}
	if len(class_counts) != 1:
		raise ValueError("All B3D trigger files must contain the same number of classes")
	num_classes = class_counts.pop()
	panel_width = loaded_stages[0][2][0].width
	panel_height = loaded_stages[0][2][0].height
	header_height = 34
	row_label_width = 62
	canvas_width = row_label_width + panel_width * len(loaded_stages)
	canvas_height = header_height + panel_height * num_classes
	contact_sheet = Image.new("RGB", (canvas_width, canvas_height), "white")
	draw = ImageDraw.Draw(contact_sheet)

	for stage_index, (label, _, _, _) in enumerate(loaded_stages):
		x = row_label_width + stage_index * panel_width
		draw.text((x + 4, 10), label, fill="black")

	csv_file = os.path.join(output_dir, "trigger-l1-comparison.csv")
	with open(csv_file, "w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(["class"] + [label for label, _, _, _ in loaded_stages])
		for class_index in range(num_classes):
			row_images = []
			writer.writerow(
				[class_index]
				+ [norms[class_index] for _, _, _, norms in loaded_stages]
			)
			y = header_height + class_index * panel_height
			draw.text((5, y + panel_height // 2), f"Class {class_index}", fill="black")
			for stage_index, (label, _, images, norms) in enumerate(loaded_stages):
				image = images[class_index]
				x = row_label_width + stage_index * panel_width
				contact_sheet.paste(image, (x, y))
				row_images.append((label, image, norms[class_index]))

			class_panel = Image.new(
				"RGB",
				(panel_width * len(row_images), panel_height + header_height),
				"white",
			)
			class_draw = ImageDraw.Draw(class_panel)
			for stage_index, (label, image, l1_norm) in enumerate(row_images):
				x = stage_index * panel_width
				class_draw.text((x + 4, 3), f"{label}\nL1={l1_norm:.1f}", fill="black")
				class_panel.paste(image, (x, header_height))
			class_panel.save(os.path.join(output_dir, f"class-{class_index:02d}.png"))

	comparison_file = os.path.join(output_dir, "comparison.png")
	contact_sheet.save(comparison_file)
	logger.info(
		"Saved B3D comparison: image=%s l1_csv=%s classes=%d",
		comparison_file, csv_file, num_classes,
	)
	return comparison_file


def parse_args():
	parser = argparse.ArgumentParser(description="Compare B3D trigger files visually.")
	parser.add_argument(
		"trigger_files", nargs="+",
		help="Trigger files in label=path form, for example Clean=weights/model-TRIGGERS.pt",
	)
	parser.add_argument("--output-dir", default="images/comparison")
	parser.add_argument("--scale", type=int, default=4)
	parser.add_argument("--log-file")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	configure_logging(args.log_file, run_name="visualize-comparison")
	parsed_stages = []
	for value in args.trigger_files:
		if "=" not in value:
			raise ValueError(f"Expected label=path, got {value!r}")
		parsed_stages.append(tuple(value.split("=", 1)))
	visualize_comparison(parsed_stages, output_dir=args.output_dir, scale=args.scale)
