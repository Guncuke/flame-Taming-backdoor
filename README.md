# flame-Taming-backdoor
Experiments are only conducted on mnist,fmnit and cifar-10 datasets.
The code is a bit sketchy, because I just want to recapitalize it briefly.
One point not mentioned in the paper is that allow_single_cluster is to be set to True.

The core code of flame's implementation algorithm is shown below
```python
  # 1. HDBSCAN余弦相似度聚类
  clients_weight_total = clients_weight_total.double()
  cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic", min_cluster_size=num_clients//2+1, min_samples=1,allow_single_cluster=True)

  # L2 = torch.norm(clients_weight_total, p=2, dim=1, keepdim=True)
  # clients_weight_total = clients_weight_total.div(L2)
  # cluster = hdbscan.HDBSCAN(min_cluster_size=num_clients//2+1, min_samples=1, allow_single_cluster=True)

  cluster.fit(clients_weight_total)

  # 2. 范数中值裁剪
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

  self.model_aggregate(weight_accumulator, num_in)

  # 4. 聚合模型添加噪声
  if self.conf['defense'] == 'flame':
    lamda = 0.000012
    for name, param in self.global_model.named_parameters():
      if 'bias' in name or 'bn' in name:
        # 不对偏置和BatchNorm的参数添加噪声
        continue
      std = lamda * med * param.data.std()
      noise = torch.normal(0, std, size=param.size()).cuda()
      param.data.add_(noise)


# 模型聚合
def model_aggregate(self, weight_accumulator, num):
	for name, data in self.global_model.state_dict().items():
		
		update_per_layer = weight_accumulator[name] / num

		if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
```
