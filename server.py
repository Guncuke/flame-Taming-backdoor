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
		
	# 在聚合前，对客户端提交上来的模型参数进行筛选
	def model_sift(self, round, clients_weight, all_candidates, true_bad, true_good):
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
		num_clients = clients_weight_total.shape[0]
		euclidean = (clients_weight_ ** 2).sum(1).sqrt()
		med = euclidean.median()
		tpr, tnr = 0, 0


		if self.conf['defense'] == 'flame':

			# # 1. HDBSCAN余弦相似度聚类
			clients_weight_total = clients_weight_total.double()
			cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic", min_cluster_size=num_clients//2+1, min_samples=1,allow_single_cluster=True)

			# L2 = torch.norm(clients_weight_total, p=2, dim=1, keepdim=True)
			# clients_weight_total = clients_weight_total.div(L2)
			# cluster = hdbscan.HDBSCAN(min_cluster_size=num_clients//2+1, min_samples=1, allow_single_cluster=True)

			cluster.fit(clients_weight_total)
			predict_good = []
			predict_bad = []
			for i, j in enumerate(cluster.labels_):
				if j == 0:
					predict_good.append(all_candidates[i])
				else:
					predict_bad.append(all_candidates[i])

			print(cluster.labels_)
			predict_good = set(predict_good)
			predict_bad = set(predict_bad)
			true_bad = set(true_bad)
			true_good = set(true_good)
			if len(true_good) == 0 and len(predict_good) == 0:
				tnr = 1
			elif len(predict_good) == 0 and len(true_good)!=0:
				tnr = 0
			else:
				tnr = len(true_good & predict_good) / len(predict_good)

			if len(true_bad) == 0 and len(predict_bad) == 0:
				tpr = 1
			elif len(predict_bad) == 0 and len(true_bad)!=0:
				tpr = 0
			else:
				tpr = len(true_bad & predict_bad) / len(predict_bad)

			# 2. 范数中值裁剪
			for i, data in enumerate(clients_weight):
				gama = med.div(euclidean[i])
				if gama > 1:
					gama = 1

				for name, params in data.items():
					params.data = (params.data * gama).to(params.data.dtype)

		elif self.conf['defense'] == 'krum':
			# 记录距离与得分
			number = 6
			if round == 4:
				number = 7
			dis = torch.zeros(num_clients, num_clients)
			score = torch.zeros(num_clients)
			for i in range(num_clients):
				for j in range(i+1, num_clients):
					dis[i][j] = torch.norm(clients_weight_total[i] - clients_weight_total[j], p=2)
					dis[j][i] = dis[i][j]

			# 获取最近的6个模型参数，包括自己
			for i, di in enumerate(dis):
				values, _ = torch.topk(di, k=number, largest=False)
				score[i] = values.sum()

			# 获得得分最低的6个模型参数
			_, indices = torch.topk(score, k=number, largest=False)
			print(indices)

		else:
			for i, data in enumerate(clients_weight):
				gama = med.div(euclidean[i])
				if gama > 1:
					gama = 1

				for name, params in data.items():
					params.data = (params.data * gama).to(params.data.dtype)
		# 3. 聚合
		num_in = 0
		for i, data in enumerate(clients_weight):
			if self.conf['defense'] == "flame":
				if cluster.labels_[i] == 0:
					num_in += 1
					for name, params in data.items():
						weight_accumulator[name].add_(params)

			elif self.conf['defense'] == "krum":
				if i in indices:
					num_in += 1
					for name, params in data.items():
						weight_accumulator[name].add_(params)

			else:
				num_in += 1
				for name, params in data.items():
					weight_accumulator[name].add_(params)

		temp = torch.tensor([])
		for name, data in self.global_model.named_parameters():
			temp = torch.cat((temp, weight_accumulator[name].reshape(-1).cpu()))

		print(temp.norm(2))

		self.model_aggregate(weight_accumulator, num_in)

		# 4. 聚合模型添加噪声

		if self.conf['defense'] == 'flame' or self.conf['defense'] == 'canyou':
			lamda = 0.000012
			for name, param in self.global_model.named_parameters():
				if 'bias' in name or 'bn' in name:
					# 不对偏置和BatchNorm的参数添加噪声
					continue
				std = lamda * med * param.data.std()
				noise = torch.normal(0, std, size=param.size()).cuda()
				param.data.add_(noise)

		return tpr, tnr

	# 模型聚合
	def model_aggregate(self, weight_accumulator, num):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] / num

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
				if self.conf["type"] == 'mnist':
					image[0][3:5, 3:5] = 2.821
				elif self.conf["type"] == "fmnist":
					image[0][3:5, 3:5] = 2.028
				else:
					image[0][3:7, 3:7] = 2.514
					image[1][3:7, 3:7] = 2.597
					image[2][3:7, 3:7] = 2.754
			
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
			# correct_poison += pred_poison.eq(target.data.view_as(pred_poison)).cpu().sum().item()

			# 要算在后门图片上的正确率，那原来就是后门标签的数据肯定不能算进来，需要去掉
			# 就比如图片标签本来就是8，而后门攻击目标也是8，如果预测出来是8这个肯定不能算入后门攻击成功
			for i in range(data.size()[0]):
				if pred_poison[i] == self.conf['poison_num'] and target[i] != self.conf['poison_num']:
					total_poison_count += 1

		# correct_poison -= total_poison_count
		correct_poison = total_poison_count
		acc = 100.0 * (float(correct) / float(dataset_size))
		acc_poison = 100.0 * (float(correct_poison) / float(dataset_size))

		total_l = total_loss / dataset_size

		return acc, acc_poison, total_l
