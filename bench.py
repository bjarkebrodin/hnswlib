#!/opt/homebrew/bin/python3
import hnswlib
import numpy as np
import os
import subprocess
from sys import argv
from time import perf_counter_ns as ns, sleep

D = 32
N = 100000
Q = 10000
K = 10
ef_construction = 100
ef_search = K*2
M = 8

# todo: check for cached data of these dimensions otherwise generate it

topk = np.fromfile("tests/cpp/data/gt.bin", dtype=np.int32).reshape(Q, K)
batch = np.fromfile("tests/cpp/data/batch_final.bin", dtype=np.float32).reshape(N, D)
queries = np.fromfile("tests/cpp/data/queries.bin", dtype=np.float32).reshape(Q, D)

hnsw_index = hnswlib.Index(space='l2', dim=D)

construction_time = ns()
hnsw_index.init_index(max_elements=N, ef_construction=ef_construction, M=M)
hnsw_index.set_ef(ef_search)
hnsw_index.set_num_threads(1)
print("Adding batch of %d elements" % (len(batch)))
hnsw_index.add_items(batch)
print("Indices built")
construction_time = ns() - construction_time

# Query the elements and measure recall:
labels_bf = topk


pid = os.getpid()
tracer = subprocess.Popen(["xctrace", "record", "--template", "CPU Profiler", "--attach", f"{pid}"], executable='xctrace')
sleep(3)
query_time = ns()
labels_hnsw, distances_hnsw = hnsw_index.knn_query(queries, K)
query_time = (ns() - query_time)
pr_point_query_time = query_time // Q

# Measure recall
correct = 0
for i in range(Q):
    for label in labels_hnsw[i]:
        for correct_label in labels_bf[i]:
            if label == correct_label:
                correct += 1
                break


print("recall          : ", float(correct)/(K*Q))
print("construction    : ", construction_time//1000, "ms")
print("query total     : ", query_time//1000, "ms")
print("query pr. point : ", pr_point_query_time, "ns")

