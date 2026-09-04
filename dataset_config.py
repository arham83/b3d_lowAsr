"""Shared dataset configuration for CIFAR-10 and GTSRB."""

from dataclasses import dataclass

import torch
import torchvision
import torchvision.transforms as transforms


@dataclass(frozen=True)
class DatasetSpec:
	name: str
	num_classes: int
	mean: tuple
	std: tuple
	image_size: int = 32
	channels: int = 3


DATASET_SPECS = {
	"cifar10": DatasetSpec(
		name="cifar10",
		num_classes=10,
		mean=(0.4914, 0.4822, 0.4465),
		std=(0.2023, 0.1994, 0.2010),
	),
	"gtsrb": DatasetSpec(
		name="gtsrb",
		num_classes=43,
		mean=(0.3403, 0.3121, 0.3214),
		std=(0.2724, 0.2608, 0.2669),
	),
}


def normalize_dataset_name(dataset_name):
	name = dataset_name.lower().replace("-", "").replace("_", "")
	if name == "cifar10":
		return "cifar10"
	if name == "gtsrb":
		return "gtsrb"
	raise ValueError(
		f"Unsupported dataset {dataset_name!r}; choose from: cifar10, gtsrb"
	)


def get_dataset_spec(dataset_name):
	return DATASET_SPECS[normalize_dataset_name(dataset_name)]


def get_transform(dataset_name, normalized):
	"""Build a transform that always returns a 32x32 tensor."""
	spec = get_dataset_spec(dataset_name)
	steps = []
	if spec.name == "gtsrb":
		steps.append(transforms.Resize((spec.image_size, spec.image_size)))
	steps.append(transforms.ToTensor())
	if normalized:
		steps.append(transforms.Normalize(spec.mean, spec.std))
	return transforms.Compose(steps)


def get_normalize(dataset_name):
	spec = get_dataset_spec(dataset_name)
	return transforms.Normalize(spec.mean, spec.std)


def load_dataset(dataset_name, train, normalized=False, root="./data", download=True):
	"""Load a CIFAR-10 or GTSRB split with a consistent tensor shape."""
	name = normalize_dataset_name(dataset_name)
	transform = get_transform(name, normalized=normalized)
	if name == "cifar10":
		return torchvision.datasets.CIFAR10(
			root=root, train=train, download=download, transform=transform
		)
	return torchvision.datasets.GTSRB(
		root=root,
		split="train" if train else "test",
		download=download,
		transform=transform,
	)




def default_checkpoint_path(name, dataset_name):
	dataset_name = normalize_dataset_name(dataset_name)
	prefix = "" if dataset_name == "cifar10" else f"{dataset_name}-"
	return f"weights/{prefix}{name}.pt"


def load_model_state(model, checkpoint_file, device):
	"""Load module or state-dict checkpoints with optional DataParallel prefixes."""
	checkpoint = torch.load(checkpoint_file, map_location=device)
	if isinstance(checkpoint, torch.nn.Module):
		state_dict = checkpoint.state_dict()
	elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
		state_dict = checkpoint["state_dict"]
	else:
		state_dict = checkpoint

	model_uses_module = next(iter(model.state_dict())).startswith("module.")
	file_uses_module = next(iter(state_dict)).startswith("module.")
	if file_uses_module and not model_uses_module:
		state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
	elif model_uses_module and not file_uses_module:
		state_dict = {"module." + key: value for key, value in state_dict.items()}
	model.load_state_dict(state_dict)
