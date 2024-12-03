#!/opt/homebrew/bin/python3

import hnswlib
import numpy as np

dim = 10
num_elements = 10000
data = np.float32(np.random.random((num_elements, dim)))

p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=100, M=16)
p.set_ef(10)
#p.set_num_threads(4)
p.add_items(data)

for i in range(10):
  labels, distances = p.knn_query(data, k=1)
  print(labels, distances)

del p
