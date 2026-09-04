import argparse

import torch

from dataset_config import (
	default_checkpoint_path, get_dataset_spec, get_normalize,
	load_dataset, load_model_state,
)
from models.resnet import ResNet18
from poison import PoisonedDataset
import masks

def g(x): return (torch.tanh(x)+1)/2

def parse_args():
	parser = argparse.ArgumentParser(description="Validate a CIFAR-10 or GTSRB model.")
	parser.add_argument("--dataset-name", choices=("cifar10", "gtsrb"), default="cifar10")
	parser.add_argument("--backdoor", type=int, choices=range(1, 11), default=5)
	parser.add_argument("--checkpoint")
	parser.add_argument("--device")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	print("Preparing datasets")
	mask, pattern, name, c = getattr(masks, f"backdoor{args.backdoor}")()
	spec = get_dataset_spec(args.dataset_name)
	file = args.checkpoint or default_checkpoint_path(name, args.dataset_name)
	
	device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

	model = ResNet18(num_classes=spec.num_classes)
	model = model.to(device)
	if device == "cuda" and torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
	load_model_state(model, file, device)
	model.eval()

	#distribution_params = torch.load("weights/poisoned-1xbottom_right_green-TRIGGERS.pt")
	#theta_m, theta_p = distribution_params[0]
	#mask = g(theta_m)>=0.5
	#pattern = g(theta_p)
	
	mask = mask.to('cpu')
	pattern = pattern.to('cpu')


	testset = load_dataset(args.dataset_name, train=False, normalized=True)
	testset_poisoned = PoisonedDataset(
		load_dataset(args.dataset_name, train=False, normalized=False),
		mask, pattern, c, get_normalize(args.dataset_name), poison_percent=1,
	)
	
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
	testloader_poisoned = torch.utils.data.DataLoader(testset_poisoned, batch_size=100, shuffle=False, num_workers=2)

	print("Evaluating...")

	correct = 0
	total = 0
	with torch.no_grad():
		count = 0
		for inputs, targets in testloader:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
		accuracy = (100.*correct/total)
		print("Accuracy on normal dataset: " + str(accuracy))

	correct = 0
	total = 0
	with torch.no_grad():
		count = 0
		for inputs, targets in testloader_poisoned:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
		accuracy = (100.*correct/total)
		print("Accuracy on poisoned dataset: " + str(accuracy))