import argparse
import logging
import os
import time
from itertools import islice

import torch
import torch.nn.functional as F
from dataset_config import (
	default_checkpoint_path,
	get_dataset_spec,
	get_normalize,
	load_dataset,
	load_model_state,
	normalize_dataset_name,
)
from models.resnet import ResNet18
from poison import poison_batched
from logging_config import configure_logging


logger = logging.getLogger(__name__)

def g(x): return (torch.tanh(x)+1)/2

def b3d(
	model, c, dataset_name="cifar10", device=None, seed=0,
	batch_size=32, samples=20, max_batches=None,
):
	dataset_name = normalize_dataset_name(dataset_name)
	spec = get_dataset_spec(dataset_name)
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	if batch_size < 1 or samples < 1:
		raise ValueError("batch_size and samples must be at least 1")
	if max_batches is not None and max_batches < 1:
		raise ValueError("max_batches must be at least 1 or None")
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	model = model.to(device)

	lambd = torch.tensor(2.5e-3, device=device, requires_grad=True)
	k = samples
	epochs = 1
	sigma = 0.1

	normalize = get_normalize(dataset_name)
	def loss(x, m, p, c, sep=False): 
		predicted = model.forward(normalize(poison_batched(x,m,p)))
		target = torch.zeros(predicted.shape).to(device)
		target[:,c] = 1
		if sep:
			return F.cross_entropy(predicted, target), lambd * torch.linalg.norm(torch.flatten(m),ord=1)
		else:
			return F.cross_entropy(predicted, target) + lambd*torch.linalg.norm(torch.flatten(m),ord=1)

	dataset = load_dataset(dataset_name, train=False, normalized=False)
	loader_generator = torch.Generator().manual_seed(seed)
	dataloader = torch.utils.data.DataLoader(
		dataset, batch_size=batch_size, shuffle=True, num_workers=2,
		generator=loader_generator,
	)

	theta_m = torch.full(size=(spec.image_size, spec.image_size),fill_value=-1.14).to(device)
	theta_p = torch.full(size=(spec.channels, spec.image_size, spec.image_size),fill_value=0.0).to(device)
	optimizer = torch.optim.Adam([
                {'params': (theta_m,), 'lr': 0.05},
                {'params': (theta_p,), 'lr': 0.05},
				{'params': (lambd,), 'lr': 0.0000005}
            ], lr=0.05)	

	best_loss = None
	best_theta_m = torch.full(size=(spec.image_size, spec.image_size),fill_value=-1.14).to(device)
	best_theta_p = torch.full(size=(spec.channels, spec.image_size, spec.image_size),fill_value=0.0).to(device)
	iter = 0
	best_iter = 0
	max_iter = min(len(dataloader), max_batches) if max_batches else len(dataloader)

	for _ in range(epochs):

		batches = dataloader if max_batches is None else islice(dataloader, max_batches)
		for inputs, _ in batches:
			inputs = inputs.to(device)
			optimizer.zero_grad()
			theta_m.grad = torch.zeros_like(theta_m).to(theta_m.device)
			theta_p.grad = torch.zeros_like(theta_p).to(theta_p.device)

			with torch.no_grad():
				for _ in range(k):
					m = torch.bernoulli(g(theta_m))
					theta_m.grad += loss(inputs, m, g(theta_p), c) * 2 * (m - g(theta_m))

				for _ in range(k):
					eps = torch.normal(mean=0, std=1, size=(spec.channels, spec.image_size, spec.image_size)).to(device)
					theta_p.grad += loss(inputs, g(theta_m), g(theta_p + sigma*eps), c) * eps

			f, l1 = loss(inputs, (g(theta_m)>=0.5).float(), g(theta_p), c, sep=True)
			l = f+l1
			
			if best_loss == None or l < best_loss:
				logger.info(
					"B3D class=%d new best: iteration=%d/%d l1=%.2f",
					c, iter, max_iter, (l1 / lambd).item(),
				)
				best_loss = l
				best_theta_m = theta_m.detach().clone()
				best_theta_p = theta_p.detach().clone()
				best_iter = iter

			theta_m.grad /= k
			theta_p.grad /= k*sigma
			l.backward()
			optimizer.step()
			iter += 1
			
	logger.info("B3D class=%d optimization complete: best_iteration=%d", c, best_iter)
	return best_theta_m, best_theta_p

def mad(triggers, anomaly_threshold=4.5):
	if anomaly_threshold <= 0:
		raise ValueError("anomaly_threshold must be greater than 0")
	l1_norms = [torch.linalg.norm(torch.flatten(m), ord=1) for m, _ in triggers]
	median = torch.median(torch.stack(l1_norms))
	deviations = [abs(median - l1) for l1 in l1_norms]
	mad_value = torch.median(torch.stack(deviations))
	if mad_value.item() == 0:
		anomaly_indexes = [
			torch.full_like(dev, float("inf")) if dev.item() else torch.zeros_like(dev)
			for dev in deviations
		]
	else:
		anomaly_indexes = [dev / (mad_value * 1.4826) for dev in deviations]

	logger.info(
		"B3D detection summary: median_l1=%.4f mad=%.4f anomaly_threshold=%.2f",
		median, mad_value, anomaly_threshold,
	)
	detected_classes = []
	for c, anomaly_index in enumerate(anomaly_indexes):
		is_backdoor = (
			l1_norms[c] < median and anomaly_index > anomaly_threshold
		) or l1_norms[c] < median / 4
		if is_backdoor:
			detected_classes.append(c)
			log = logger.warning
			status = "BACKDOOR"
		else:
			log = logger.info
			status = "clean"
		log(
			"B3D result: class=%d l1=%.2f deviation=%.2f "
			"mad_score=%.2f anomaly_index=%.2f status=%s",
			c, l1_norms[c], deviations[c], anomaly_index, anomaly_index, status,
		)
	logger.info("B3D detected backdoor classes: %s", detected_classes or "none")
	return detected_classes


def b3d_complete(
	name, dataset_name="cifar10", checkpoint_file=None, device=None,
	anomaly_threshold=4.5, seed=0, batch_size=32, samples=20,
	max_batches=None,
):
	dataset_name = normalize_dataset_name(dataset_name)
	spec = get_dataset_spec(dataset_name)
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	logger.info(
		"Starting B3D detection: model=%s dataset=%s device=%s seed=%d "
		"anomaly_threshold=%.2f batch_size=%d samples=%d max_batches=%s",
		name, dataset_name, device, seed, anomaly_threshold, batch_size,
		samples, max_batches or "all",
	)
	start_time = time.time()
	weights_file = checkpoint_file or default_checkpoint_path(name, dataset_name)
	save_location = os.path.splitext(weights_file)[0] + "-TRIGGERS.pt"

	model = ResNet18(num_classes=spec.num_classes).to(device)
	if device.startswith("cuda") and torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
	load_model_state(model, weights_file, device)
	logger.info("Loaded checkpoint for B3D: %s", weights_file)
	model.eval()

	distribution_params = []
	for c in range(spec.num_classes):
		logger.info(
			"Scanning B3D class %d/%d | elapsed=%.2f min",
			c + 1, spec.num_classes, (time.time() - start_time) / 60,
		)
		theta_m_c, theta_p_c = b3d(
			model, c, dataset_name=dataset_name, device=device, seed=seed + c,
			batch_size=batch_size, samples=samples, max_batches=max_batches,
		)
		distribution_params.append((theta_m_c, theta_p_c))
		torch.save(distribution_params, save_location)
		logger.info("Saved B3D trigger progress: %s", save_location)

	triggers = [
		((g(theta_m) >= 0.5).float(), g(theta_p))
		for theta_m, theta_p in distribution_params
	]
	detected_classes = mad(triggers, anomaly_threshold=anomaly_threshold)
	logger.info(
		"B3D detection complete: model=%s detected_classes=%s elapsed=%.2f min",
		name, detected_classes or "none", (time.time() - start_time) / 60,
	)
	return detected_classes


def parse_args():
	parser = argparse.ArgumentParser(description="Run B3D on a CIFAR-10 or GTSRB model.")
	parser.add_argument("name", nargs="?", default="backdoored-1-reversed")
	parser.add_argument("--dataset-name", choices=("cifar10", "gtsrb"), default="cifar10")
	parser.add_argument("--checkpoint")
	parser.add_argument("--anomaly-threshold", type=float, default=4.5)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--samples", type=int, default=20)
	parser.add_argument("--max-batches", type=int, default=0, help="Batches per class; use 0 for all")
	parser.add_argument("--device")
	parser.add_argument("--log-file", help="Log file path (default: logs/b3d-*.log)")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	configure_logging(args.log_file, run_name=f"b3d-{args.dataset_name}-{args.name}")
	b3d_complete(
		args.name, dataset_name=args.dataset_name,
		checkpoint_file=args.checkpoint, device=args.device,
		anomaly_threshold=args.anomaly_threshold, seed=args.seed,
		batch_size=args.batch_size, samples=args.samples,
		max_batches=args.max_batches or None,
	)
