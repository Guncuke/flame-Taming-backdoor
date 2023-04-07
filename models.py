import torch 
from torchvision import models
import torch.nn as nn
import json


class Model:

	def __init__(self, name):
		with open('./utils/conf.json', 'r') as f:
			conf = json.load(f)

		if name == "resnet18":
			model = models.resnet18()
			if conf['type'] == 'mnist' or conf['type'] == 'fmnist':
				model.conv1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
				num_ftrs = model.fc.in_features
				model.fc = nn.Linear(num_ftrs, 10)
				model.load_state_dict(torch.load('./data/resnet18.pkl'))
			elif conf['type'] == 'cifar':
				model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
				model.maxpool = nn.MaxPool2d(1, 1, 0)  # 通过1x1的池化核让池化层失效
				num_ftrs = model.fc.in_features
				model.fc = nn.Linear(num_ftrs, 10)

		elif name == 'logistic':
			class Model(nn.Module):
				def __init__(self):
					super(Model, self).__init__()
					self.nec = torch.nn.Sequential(
						torch.nn.Linear(784, 512),
						torch.nn.Sigmoid(),
						torch.nn.Linear(512, 10))

				def forward(self, x):
					x = x.view(-1, 784)
					y = self.nec(x)
					return y
			model = Model()

		# TODO: 多尝试几个模型
		elif name == "resnet50":
			model = models.resnet50()
		elif name == "densenet121":
			model = models.densenet121()
		elif name == "alexnet":
			model = models.alexnet()
		elif name == "vgg16":
			model = models.vgg16()
		elif name == "vgg19":
			model = models.vgg19()
		elif name == "googlenet":
			model = models.googlenet()

		if torch.cuda.is_available():
			self.model = model.cuda()
		else:
			self.model = model


