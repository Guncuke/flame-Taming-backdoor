import torch
from models import Model
import hdbscan


class Server(Model):
	
	def __init__(self, conf, eval_dataset):

		super().__init__(conf['model_name'])
	
		self.conf = conf

		self.global_model = self.model
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

	# 分发模型
	def model_distribution(self, candidates):
		for c in candidates:
			for name, param in self.global_model.state_dict().items():
				c.local_model.state_dict()[name].copy_(param.clone())
		
	# TODO: 在聚合前，对客户端提交上来的模型参数进行筛选
	def model_sift(self, clients_weight):
		# 用来存储筛选后模型参数和
		weight_accumulator = {}
		for name, params in self.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)

		# 0. 数据预处理，将clients_weight展开成二维tensor, 方便聚类计算
		clients_weight_ = []
		clients_weight_total = []
		for data in clients_weight:
			client_weight = torch.tensor([])
			client_weight_total = torch.tensor([])

			for name, params in data.items():
				client_weight = torch.cat((client_weight, params.reshape(-1).cpu()))
				if name == 'fc.weight' or name == 'fc.bias':
					client_weight_total = torch.cat((client_weight_total, (params + self.global_model.state_dict()[name]).reshape(-1).cpu()))

			clients_weight_.append(client_weight)
			clients_weight_total.append(client_weight_total)

		# 获得了每个客户端模型的参数，矩阵大小为(客户端数, 参数个数)
		clients_weight_ = torch.stack(clients_weight_)
		clients_weight_total = torch.stack(clients_weight_total)
		# 1. HDBSCAN余弦相似度聚类
		num_clients = clients_weight_total.shape[0]
		clients_weight_total = clients_weight_total.double()
		cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic", min_cluster_size=num_clients//2, min_samples=1)

		# L2 = torch.norm(clients_weight_total, p=2, dim=1, keepdim=True)
		# clients_weight_total = clients_weight_total.div(L2)
		# cluster = hdbscan.HDBSCAN(min_cluster_size=num_clients//2, min_samples=1)
		cluster.fit(clients_weight_total)
		print(cluster.labels_)

		# 2. 范数中值裁剪
		euclidean = (clients_weight_**2).sum(1).sqrt()
		med = euclidean.median()
		for i, data in enumerate(clients_weight):
			gama = med.div(euclidean[i])
			if gama > 1:
				gama = 1

			for name, params in data.items():
				params.data = (params.data * gama).to(params.data.dtype)

		# 3. 聚合
		for data in clients_weight:
			for name, params in data.items():
				weight_accumulator[name].add_(params)

		self.model_aggregate(weight_accumulator)

		# # 4. TODO:聚合模型添加噪声 no test yet!
		# epsilon = 0.01
		# delta = 1/num_clients
		# sigma = med.div(epsilon) * torch.sqrt(2 * torch.log(torch.tensor(1.25/delta)))
		# # print(sigma)
		#
		# for name, data in self.global_model.state_dict().items():
		# 	data.add_(torch.normal(0, sigma).to(data.dtype))

	# 模型聚合
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]

			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
	
	# 模型评估
	def model_eval(self):

		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		correct_poison = 0
		dataset_size = 0
		total_poison_count = 0

		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 
			dataset_size += data.size()[0]

			poison_data = data.clone()

			for i, image in enumerate(poison_data):
				image[0][3:5, 3:5] = 2.821
			
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
				poison_data = poison_data.cuda()
				
			output = self.global_model(data)
			output_poison = self.global_model(poison_data)
			
			total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
			
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			pred_poison = output_poison.data.max(1)[1]  # 在后门图片上的预测值

			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
			correct_poison += pred_poison.eq(target.data.view_as(pred_poison)).cpu().sum().item()

			# 要算在后门图片上的正确率，那原来就是后门标签的数据肯定不能算进来，需要去掉
			# 就比如图片标签本来就是8，而后门攻击目标也是8，如果预测出来是8这个肯定不能算入后门攻击成功
			for i in range(data.size()[0]):
				if pred_poison[i] == self.conf['poison_num'] and target[i] == self.conf['poison_num']:
					total_poison_count += 1

		correct_poison -= total_poison_count
		acc = 100.0 * (float(correct) / float(dataset_size))
		acc_poison = 100.0 * (float(correct_poison) / (float(dataset_size) - total_poison_count))

		total_l = total_loss / dataset_size

		return acc, acc_poison, total_l
