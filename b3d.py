import logging
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

from models.resnet import ResNet18
from poison import poison_batched
from logging_config import configure_logging


logger = logging.getLogger(__name__)

def g(x): return (torch.tanh(x)+1)/2

def b3d(model, c):
	device = "cuda"

	model = model.to(device)

	lambd = torch.tensor(2.5e-3, requires_grad=True)
	k = 20
	epochs = 1
	sigma = 0.1
	batch_size = 32

	normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	def loss(x, m, p, c, sep=False): 
		predicted = model.forward(normalize(poison_batched(x,m,p)))
		target = torch.zeros(predicted.shape).to(device)
		target[:,c] = 1
		if sep:
			return F.cross_entropy(predicted, target), lambd * torch.linalg.norm(torch.flatten(m),ord=1)
		else:
			return F.cross_entropy(predicted, target) + lambd*torch.linalg.norm(torch.flatten(m),ord=1)

	transform = transforms.Compose([transforms.ToTensor()])
	dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

	theta_m = torch.full(size=(32,32),fill_value=-1.14).to("cuda")
	theta_p = torch.full(size=(3,32,32),fill_value=0.0).to("cuda")
	optimizer = torch.optim.Adam([
                {'params': (theta_m,), 'lr': 0.05},
                {'params': (theta_p,), 'lr': 0.05},
				{'params': (lambd,), 'lr': 0.0000005}
            ], lr=0.05)	

	best_loss = None
	best_theta_m = torch.full(size=(32,32),fill_value=-1.14).to("cuda")
	best_theta_p = torch.full(size=(3,32,32),fill_value=0.0).to("cuda")
	iter = 0
	best_iter = 0
	max_iter = len(dataloader)

	for _ in range(epochs):

		for inputs, _ in dataloader:
			inputs = inputs.to(device)
			optimizer.zero_grad()
			theta_m.grad = torch.zeros((32,32)).to(theta_m.device)
			theta_p.grad = torch.zeros((3,32,32)).to(theta_p.device)

			with torch.no_grad():
				for _ in range(k):
					m = torch.bernoulli(g(theta_m))
					theta_m.grad += loss(inputs, m, g(theta_p), c) * 2 * (m - g(theta_m))

				for _ in range(k):
					eps = torch.normal(mean=0, std=1, size=(3,32,32)).to(device)
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
			
	return best_theta_m, best_theta_p

def mad(triggers):
	l1_norms = [torch.linalg.norm(torch.flatten(m),ord=1) for m, _ in triggers]
	median = torch.median(torch.tensor(l1_norms))

	deviations = [abs(median-l1) for l1 in l1_norms]
	MAD = torch.median(torch.tensor(deviations))
	AIs = [dev/(MAD*1.4826) for dev in deviations]

	logger.info(
		"B3D detection summary: median_l1=%.4f mad=%.4f",
		median.item(), MAD.item(),
	)
	for c in range(len(AIs)):
		if (l1_norms[c] < median and AIs[c] > 2) or (l1_norms[c] < median/4):
			logger.warning(
				"B3D result: class=%d l1=%.2f deviation=%.2f anomaly_index=%.2f status=BACKDOOR",
				c, l1_norms[c].item(), deviations[c].item(), AIs[c].item(),
			)
		else:
			logger.info(
				"B3D result: class=%d l1=%.2f deviation=%.2f anomaly_index=%.2f",
				c, l1_norms[c].item(), deviations[c].item(), AIs[c].item(),
			)


def b3d_complete(name):
	logger.info("Starting B3D: model=%s", name)
	start_time = time.time()

	weights_file = "weights/" + name + ".pt"
	save_location = "weights/" + name + "-TRIGGERS.pt"

	model = ResNet18()
	model = torch.nn.DataParallel(model)
	model.load_state_dict(torch.load(weights_file))
	model.eval()

	distribution_params = []
	for c in range(10):
		logger.info(
			"Scanning B3D class %d/10 | elapsed=%.2f min",
			c + 1, (time.time() - start_time) / 60,
		)

		theta_m_c, theta_p_c = b3d(model, c)
		distribution_params.append((theta_m_c, theta_p_c))
		torch.save(distribution_params, save_location)
		logger.info("Saved B3D trigger progress: %s", save_location)
	
	triggers = [((g(theta_m)>=0.5).float(), g(theta_p)) for theta_m, theta_p in distribution_params]
	mad(triggers)
	logger.info(
		"B3D complete: model=%s triggers=%s elapsed=%.2f min",
		name, save_location, (time.time() - start_time) / 60,
	)
	

if __name__=="__main__":
	configure_logging(run_name="cifar10-b3d-clean")
	b3d_complete("clean")
	# b3d_complete("backdoored-1")
	# b3d_complete("backdoored-1-reversed")
