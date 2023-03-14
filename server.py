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
		clients_weight = clients_weight.double()
		num_clients = clients_weight.shape[0]
		cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic", min_cluster_size=2, min_samples=1)
		# L2 = torch.norm(clients_weight, p=2, dim=1, keepdim=True)
		# clients_weight = clients_weight.div(L2)
		cluster.fit(clients_weight)
		print(cluster.labels_)


	# 模型聚合 fedavg
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
