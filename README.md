# flame-Taming-backdoor
Experiments are only conducted on mnist,fmnit and cifar-10 datasets.
The code is a bit sketchy, because I just want to recapitalize it briefly.
One point not mentioned in the paper is that allow_single_cluster is to be set to True.

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
```
