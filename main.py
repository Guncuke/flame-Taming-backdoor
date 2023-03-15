import json
from plot import plot_E
import random
from torch.utils.data import Subset
from server import *
from client import *
import datasets
import copy
import torch


if __name__ == '__main__':

	with open('./utils/conf.json', 'r') as f:
		conf = json.load(f)	
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	# 实例化服务器
	server = Server(conf, eval_datasets)

	# 全部客户端列表
	clients = []
	random.seed(5)

	# 实例化每一个客户端
	for c in range(conf["no_models"]):

		malicious = False

		data_len = int(len(train_datasets) / conf['no_models'])

		subset_indices = range(c * data_len, (c + 1) * data_len)

		if c in conf['malicious_user']:
			malicious = True
			# 加入后门数据
			if conf['type'] == 'mnist' or conf['type'] == 'fmnist': 
				for i in range(c * data_len, c * data_len + data_len // 11):
					train_datasets.data[i][3:5, 3:5].fill_(255)
					train_datasets.targets[i].copy_(torch.tensor(conf['poison_num']))

		subset_indices = random.choices(subset_indices, k=data_len)
		subset_dataset = Subset(train_datasets, subset_indices)

		if c in conf['malicious_user']:
			plot_E.plot_image(subset_dataset)

		client_model = copy.deepcopy(server.global_model)

		clients.append(Client(conf, client_model, subset_dataset, c, malicious))
		
	accuracy = []
	accuracy_poison = []
	losses = []
	round_poison = []

	for e in range(conf["global_epochs"]):

		candidates = random.sample(clients, conf["k"])

		# 为选中的参与者分配新的模型权重
		server.model_distribution(candidates)
		
		# 客户端训练 clients_weight记录每个客户端的模型参数
		clients_weight = []
		for c in candidates:

			client_weight = {}

			# 本轮有恶意用户
			if c.client_id in conf['malicious_user']:
				round_poison.append(e)
			
			diff = c.local_train()
			# 将每个客户端的差值保存，用于之后的防御算法处理
			for name, params in server.global_model.state_dict().items():

				client_weight[name] = diff[name]

			clients_weight.append(client_weight)

		# 服务器筛选良性客户端，并将聚合后的计算结果返回
		weight_accumulator = server.model_sift(clients_weight)

		# TODO：筛选完后再加入weight_accumulator
		# 模型聚合
		server.model_aggregate(weight_accumulator)
		
		acc, acc_poison, loss = server.model_eval()
		accuracy.append(acc)
		accuracy_poison.append(acc_poison)
		losses.append(loss)

		print("Epoch %d, acc: %f, loss: %f, acc_on_poison: %f\n" % (e, acc, loss, acc_poison))
	
	plot_E.plot_loss_accuracy(losses, accuracy, round_poison, accuracy_poison)
