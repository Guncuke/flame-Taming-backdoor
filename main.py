import json
from plot import plot_E
import random
from torch.utils.data import Subset
from server import *
from client import *
import datasets
import copy
import torch
from torch.distributions.dirichlet import Dirichlet
import numpy as np


np.random.seed(42)

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # (K, ...) 记录K个类别对应的样本索引集合

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应的样本索引集合
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

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
	data_len = int(len(train_datasets) / conf['no_models'])
	if conf["data_distribution"] == 'iid':
		data_len = int(len(train_datasets) / conf['no_models'])
		for c in range(conf["no_models"]):
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
			client_model = copy.deepcopy(server.global_model)
			clients.append(Client(conf, client_model, subset_dataset, c))

	elif conf["data_distribution"] == 'non_iid':
		N_CLIENTS = conf["no_models"]
		DIRICHLET_ALPHA = 0.5

		input_sz, num_cls = train_datasets.data[0].shape[0], len(train_datasets.classes)

		train_labels = np.array(train_datasets.targets)

		# 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
		client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

		for c, subset_indices in enumerate(client_idcs):
			subset_dataset = Subset(train_datasets, subset_indices)
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
			print(tpr, tnr)

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
