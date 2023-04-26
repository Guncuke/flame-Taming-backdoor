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

		data_len = int(len(train_datasets) / conf['no_models'])
		if conf["data_distribution"] == 'iid':
			subset_indices = range(c * data_len, (c + 1) * data_len)

			# if c in conf['malicious_user']:
			# 	malicious = True
			# 	# 加入后门数据
			# 	if conf['type'] == 'mnist' or conf['type'] == 'fmnist':
			# 		for i in range(c * data_len, c * data_len + data_len // 5):
			# 			train_datasets.data[i][3:5, 3:5].fill_(255)
			# 			train_datasets.targets[i].copy_(torch.tensor(conf['poison_num']))

			subset_indices = random.choices(subset_indices, k=data_len)
			subset_dataset = Subset(train_datasets, subset_indices)
		elif conf["data_distribution"] == 'non_iid':
			labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

			subset_indices = torch.where(train_datasets.targets == labels[c])[0]
			subset_dataset = Subset(train_datasets, subset_indices)

		# if c in conf['malicious_user']:
			# 	plot_E.plot_image(subset_dataset)

		client_model = copy.deepcopy(server.global_model)

		clients.append(Client(conf, client_model, subset_dataset, c))
		
	accuracy = []
	accuracy_poison = []
	losses = []
	tprs = []
	tnrs = []

	lr = conf["lr"]
	for e in range(conf["global_epochs"]):
		true_malicous = []
		true_good = list(range(conf["no_models"]))
		candidates = random.sample(clients, conf["k"])
		all_candidates = []

		if e in conf['malicious_round']:
			# 随机产生1~conf['malicious_num']个后门用户
			malicious_candidates = random.sample(candidates, conf['malicious_num'])
			for malicious_candidate in malicious_candidates:
				malicious_candidate.is_poison = True
				true_malicous.append(malicious_candidate.client_id)
				true_good.remove(malicious_candidate.client_id)

		# 为选中的参与者分配新的模型权重
		server.model_distribution(candidates)

		# 客户端训练 clients_weight记录每个客户端的模型参数
		clients_weight = []
		for i, c in enumerate(candidates):

			all_candidates.append(c.client_id)
			print(f'{i}:', end='')
			diff = c.local_train(lr)

			clients_weight.append(diff)

		# 服务器筛选良性客户端，并将聚合后的计算结果返回
		tpr, tnr = server.model_sift(e, clients_weight, all_candidates, true_malicous, true_good)
		if e in conf['malicious_round']:
			tprs.append(tpr)
			tnrs.append(tnr)

		acc, acc_poison, loss = server.model_eval()
		accuracy.append(acc)
		accuracy_poison.append(acc_poison)
		losses.append(loss)

		print("Epoch %d, acc: %f, loss: %f, acc_on_poison: %f\n" % (e, acc, loss, acc_poison))

	tpr_num = sum(tprs)/len(tprs)
	tnr_num = sum(tnrs)/len(tnrs)
	print(tprs)
	print(tnrs)
	print('tpr:', tpr_num, 'tnr:', tnr_num)
	save_data = {'losses': losses, 'acc': accuracy, 'acc_poison': accuracy_poison}
	torch.save(save_data, f"{conf['type']}_{conf['malicious_num']}.pth")
	plot_E.plot_loss_accuracy(losses, accuracy, conf['malicious_round'], accuracy_poison)
