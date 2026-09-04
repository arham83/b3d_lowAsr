"""Shared terminal and file logging configuration."""

import logging
import os
from datetime import datetime


def configure_logging(log_file=None, run_name="b3d"):
	"""Configure application logging and return the absolute log-file path."""
	if log_file is None:
		timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
		log_file = os.path.join("logs", f"{run_name}-{timestamp}.log")

	log_file = os.path.abspath(log_file)
	os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
	formatter = logging.Formatter(
		"%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	file_handler = logging.FileHandler(log_file, encoding="utf-8")
	file_handler.setFormatter(formatter)

	logging.basicConfig(
		level=logging.INFO,
		handlers=[console_handler, file_handler],
		force=True,
	)
	logging.getLogger(__name__).info("Logging to %s", log_file)
	return log_file
