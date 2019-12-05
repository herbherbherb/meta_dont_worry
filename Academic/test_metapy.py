import metapy
import numpy as np
import json
import collections
from pl2 import load_ranker
from pl2 import PL2

idx = metapy.index.make_inverted_index('config.toml')
ev = metapy.index.IREval('config.toml')

with open("mapping_rev.json",'r') as file:
	meta_idx_to_given_idx =  json.load(file)


# bm25 = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500.0)
# dirich = metapy.index.DirichletPrior(mu=0.05)
# jm = metapy.index.JelinekMercer()
# pl2 = PL2(1)
# # rankers = [bm25, dirich, jm, pl2]
# ranker = metapy.index.DirichletPrior(mu=1)
# num_results = 100
# output = ''
# with open('test_queries.txt') as query_file:
# 	for query_num, line in enumerate(query_file):
# 		query = metapy.index.Document()
# 		query.content(line.strip())
# 		results = ranker.score(idx, query, num_results)
# 		for doc_id, score in results:
# 			if str(doc_id) not in meta_idx_to_given_idx: 
# 				print(str(doc_id))
# 				continue
# 			actual_id = meta_idx_to_given_idx[str(doc_id)]
# 			output += str(query_num) + '\t' + str(actual_id) + '\t' + str(score) + '\n'
# 	output = output.strip('\n')
# 	query_file.close()

# writting_file = open('Academic_domain_results.txt','w')
# writting_file.write(output)
# writting_file.close()


bm25 = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500.0)
dirich = metapy.index.DirichletPrior(mu=0.05)
jm = metapy.index.JelinekMercer()
pl2 = PL2(1)
rankers = [bm25, dirich, jm, pl2]
ranker = rankers[1]
num_results = 100
output = ''
total = 0.0
with open('train_queries.txt') as query_file:
	for query_num, line in enumerate(query_file):
		query = metapy.index.Document()
		query.content(line.strip())
		results = ranker.score(idx, query, num_results)
		avg_p = ev.avg_p(results, query_num, num_results)
		total += avg_p
	query_file.close()

print("Total average precision: {}".format(total))