#!/opt/homebrew/bin/python3
import hnswlib
import numpy as np
from sys import argv
from time import perf_counter_ns as ns

dim = 32
num_elements = int(argv[1])
k = 10
nun_queries = 1000
ef_construction = 200
ef_search = 50
M = 16

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

topk = np.fromfile("tests/cpp/data/gt.bin", dtype=np.int32)
batch = np.fromfile("tests/cpp/data/batch_final.bin", dtype=np.float32).reshape(8, 10**5)
queries = np.fromfile("tests/cpp/data/queries.bin", dtype=np.float32).reshape(8, 10**3)

print(topk)
print(queries)
print(batch)

# Declaring index
hnsw_index = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
bf_index = hnswlib.BFIndex(space='l2', dim=dim)

# Initing both hnsw and brute force indices
# max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
# during insertion of an element.
# The capacity can be increased by saving/loading the index, see below.
#
# hnsw construction params:
# ef_construction - controls index search speed/build speed tradeoff
#
# M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
# Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

construction_time = ns()
hnsw_index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
construction_time = ns() - construction_time

bf_index.init_index(max_elements=num_elements)

# Controlling the recall for hnsw by setting ef:
# higher ef leads to better accuracy, but slower search
hnsw_index.set_ef(ef_search)

# Set number of threads used during batch search/construction in hnsw
# By default using all available cores
hnsw_index.set_num_threads(1)

print("Adding batch of %d elements" % (len(data)))
hnsw_index.add_items(data)
bf_index.add_items(data)

print("Indices built")

# Generating query data
query_data = np.float32(np.random.random((nun_queries, dim)))

# Query the elements and measure recall:
labels_bf, distances_bf = bf_index.knn_query(query_data, k)


# Start profiling

query_time = ns()
labels_hnsw, distances_hnsw = hnsw_index.knn_query(query_data, k)
query_time = (ns() - query_time)
pr_point_query_time = query_time // nun_queries

# Measure recall
correct = 0
for i in range(nun_queries):
    for label in labels_hnsw[i]:
        for correct_label in labels_bf[i]:
            if label == correct_label:
                correct += 1
                break


print("recall          : ", float(correct)/(k*nun_queries))
print("construction    : ", construction_time//1000, "ms")
print("query total     : ", query_time//1000, "ms")
print("query pr. point : ", pr_point_query_time, "ns")
