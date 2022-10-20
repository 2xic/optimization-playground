from model_torch_optimizer import TorchLearning2LearnOptimizer
import torch
import argparse
from optimization_utils.logging.EpochRuns import EpochRuns


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--trained", type=str, default="yes")
parser.add_argument("-o", "--optimizer", type=str, choices=["l2l", "adam"], default="l2l")

args = parser.parse_args()

suffix = "trained" if len(args.trained) and args.optimizer =="l2l" else ""
name = args.optimizer + suffix

epoch_information = EpochRuns(name)

x = torch.tensor(.0, requires_grad=True)
y = (x-2)**2

optimizer = TorchLearning2LearnOptimizer([x], lr=1e-3, use_trained=(len(args.trained) > 0)) \
			if args.optimizer == 'l2l'\
			else \
			torch.optim.Adam([x], lr=1e-3)

for i in range(1_000):
	optimizer.zero_grad()
	y = (x-2)**2
	y.backward(retain_graph=True)
	optimizer.step()

	if i % 50 == 0:
		epoch_information.log("x", x.item(), i)
epoch_information.store()
print(x)
