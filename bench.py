import hnswlib
import numpy as np
import os
import subprocess
import psutil
from sys import argv
from time import perf_counter_ns as ns, sleep
import psutil
import math

precision = int(input('Precision (EF): '))
trace = input('Trace? [y/n]').lower().startswith('y')

D = 32
N = 100000
Q = 10000
K = 10
ef_search = precision
M = 4 * int(math.ceil(math.log(N)))
ef_construction = 10 * precision
print(f"M={M}, EF={ef_search}, EFC={ef_construction}")

topk = np.fromfile("tests/cpp/data/gt.bin", dtype=np.int32).reshape(Q, K)
batch = np.fromfile("tests/cpp/data/batch_final.bin", dtype=np.float32).reshape(N, D)
queries = np.fromfile("tests/cpp/data/queries.bin", dtype=np.float32).reshape(Q, D)

# todo: measure index size approximately
index_size = psutil.Process().memory_info().rss
hnsw_index = hnswlib.Index(space='l2', dim=D)

construction_time = ns()
hnsw_index.init_index(max_elements=N, ef_construction=ef_construction, M=M)
hnsw_index.set_ef(ef_search)
hnsw_index.set_num_threads(1)
print("Adding batch of %d elements" % (len(batch)))
hnsw_index.add_items(batch)
print("Indices built")
construction_time = ns() - construction_time
index_size = psutil.Process().memory_info().rss - index_size

# Query the elements and measure recall:
labels_bf = topk

if trace:
    pid = os.getpid()
    tracer = subprocess.Popen(["xctrace", "record", "--template", "CPU Profiler", "--attach", f"{pid}"], executable='xctrace')
    sleep(2)

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


print(f"recall@{K}            : ", f"{(float(correct)/(K*Q)):.3f}")
print( "construction         : ", construction_time//1000000000, "s")
print( "const pr.point       : ", construction_time//(1000*N), "ys")
print( "query total          : ", query_time//1000000, "ms")
print( "~query pr.point      : ", f"{pr_point_query_time/Q:.3f}", "ns")
print( "~index size          : ", index_size//1000000, "MB")
print( "~index size pr.point : ", index_size//N, "B")

