import torch
import random


class Client(object):

	def __init__(self, conf, model, train_dataset, id=-1, is_poison=False):
		
		self.conf = conf
		# 用户是否是恶意用户
		self.is_poison = is_poison
		
		self.local_model = model
		
		self.client_id = id

		self.train_dataset = train_dataset

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], shuffle=True)

	# 模型投毒(无范数裁剪)
	def model_poison(self, diff):

		for name, data in diff.items():

			num = torch.tensor(self.conf['lambda'], dtype=torch.float32)

			data.data = data.data.div(num).to(data.data.dtype)

	def local_train(self, lr):

		# 记录下训练前的模型参数
		pre_model = {}
		for name, param in self.local_model.state_dict().items():
			pre_model[name] = param.clone()

		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=self.conf['momentum'])

		epoch = self.conf['local_epochs']

		self.local_model.train()
		for _ in range(epoch):
			for _, batch in enumerate(self.train_loader):
				data, target = batch

				if self.is_poison:
					poison_data = random.sample(range(len(data)), len(data)//11)
					for index in poison_data:
						if self.conf["type"] == 'mnist':
							data[index][0][3:5, 3:5].fill_(2.821)
						elif self.conf["type"] == "fmnist":
							data[index][0][3:5, 3:5].fill_(2.028)
						else:
							data[index][0][3:7, 3:7].fill_(2.514)
							data[index][1][3:7, 3:7].fill_(2.597)
							data[index][2][3:7, 3:7].fill_(2.754)
						target[index].copy_(torch.tensor(self.conf['poison_num']))
				
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
				optimizer.step()
				
		print(f"{self.client_id} complete!", end='')
		
		if self.is_poison:
			print(f"    user{self.client_id} is malicious user!")
		else:
			print('')
		
		diff = dict()

		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - pre_model[name])

		if self.is_poison:
		  self.model_poison(diff)

		self.is_poison = False

		return diff
