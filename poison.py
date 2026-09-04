import torch
import random

def poison_batched(img_batch, mask, pattern):
	"""Apply one trigger to a batch using broadcasted tensor operations."""
	mask = mask.to(device=img_batch.device, dtype=img_batch.dtype)
	pattern = pattern.to(device=img_batch.device, dtype=img_batch.dtype)
	return (1 - mask) * img_batch + mask * pattern

def poison(img, mask, pattern):
	return (1-mask.float()) * img + mask.float() * pattern

class PoisonedDataset(torch.utils.data.Dataset):

	def __init__(
		self, dataset, mask, pattern, target_class, transform,
		poison_percent=0.1, seed=0,
	):
		self.dataset = dataset
		self.mask = mask
		self.pattern = pattern
		self.target_class = target_class
		self.transform = transform
		self.poison_percent = poison_percent

		count = int(self.poison_percent * len(dataset))
		rng = random.Random(seed)
		self.poisoned_indexes = set(rng.sample(range(len(dataset)), count))

	def __getitem__(self, id):
		if id in self.poisoned_indexes:
			features = poison(self.dataset[id][0], self.mask, self.pattern)
			return (self.transform(features), self.target_class)
		else:
			features, label = self.dataset[id]
			return (self.transform(features), label)

	def __len__(self):
		return len(self.dataset)


# Backwards-compatible name for older imports.
CIFAR10_POISONED = PoisonedDataset