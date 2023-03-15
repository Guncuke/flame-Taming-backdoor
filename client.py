import torch


class Client(object):

	def __init__(self, conf, model, train_dataset, id=-1, is_poison=False):
		
		self.conf = conf
		# 用户是否是恶意用户
		self.is_poison = is_poison
		
		self.local_model = model
		
		self.client_id = id

		self.train_dataset = train_dataset

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"])

	# 模型投毒(无范数裁剪)
	def model_poison(self, diff):

		for name, data in self.local_model.state_dict().items():

			num = torch.tensor(self.conf['lambda'], dtype=torch.float32)

			if diff[name].type() != num.type():
				diff[name] = diff[name].div(num).to(diff[name].dtype)
			else:
				diff[name].div_(num)

	def local_train(self):

		# 记录下训练前的模型参数
		pre_model = {}
		for name, param in self.local_model.state_dict().items():
			pre_model[name] = param.clone()

		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])

		epoch = self.conf['local_epochs']

		self.local_model.train()
		for _ in range(epoch):
			for _, batch in enumerate(self.train_loader):
				data, target = batch
				
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

		return diff
