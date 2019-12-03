import metapy
import numpy as np
import os
import json
import re
import traceback
import xml.etree.ElementTree as ET
import collections
from pl2 import load_ranker
from pl2 import PL2

idx = metapy.index.make_inverted_index('config.toml')
# ranker = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500.0)
# ranker = metapy.index.DirichletPrior(mu=100)
# ranker = metapy.index.JelinekMercer()
# ranker = PL2(1)
# fidx = metapy.index.make_forward_index('./config.toml')
# dset = metapy.learn.Dataset(fidx)
ev = metapy.index.IREval('config.toml')


# line_number = 0
# dic = collections.defaultdict(lambda: 0)
# reverse = collections.defaultdict(lambda: 0)
# with open('./test/test.dat', 'r') as f:
# 	line  = f.readline()
# 	actual_idx = line.split("   ")[0].split("<")[0]
# 	dic[actual_idx] = line_number
# 	reverse[line_number] = actual_idx
# 	while line:
# 		line_number += 1
# 		line = f.readline()
# 		if not line: break
# 		actual_idx = line.split("   ")[0].split("<")[0]
# 		dic[actual_idx] = line_number
# 		reverse[line_number] = actual_idx
# with open('given_idx_to_meta_idx.json', 'w+') as fp:
# 	json.dump(dic, fp)
# with open('meta_idx_to_given_idx.json', 'w+') as fp:
# 	json.dump(reverse, fp)

# res = []
# with open('given_idx_to_meta_idx.json', 'r') as fp:
# 	dic = json.load(fp)
# filepath = './train_qrel.txt'
# with open(filepath) as fp: 
# 	for cnt, line in enumerate(fp):
# 		ln, idx, rel = line.strip().split(" ")
# 		if idx in dic:
# 		# print(ln, idx, rel)
# 			res.append(str(ln) + " " + str(dic[idx]) + " " + str(rel))
# 	print(len(res))

# with open('train_qrel_converted.txt', 'w+') as filehandle:
#     for listitem in res:
#         filehandle.write('%s\n' % listitem)


# meta_idx_to_given_idx = None
# with open('meta_idx_to_given_idx.json', 'r') as fp:
# 	meta_idx_to_given_idx = json.load(fp)
# ret = ''
# total = 0.0
# num_results = 100
# with open('train_input.txt') as query_file:
# 	for query_num, line in enumerate(query_file):
# 		query.content(line.strip())
# 		results = ranker.score(idx, query, num_results)
# 		# print(results)
# 		for doc_id, score in results:
# 			doc_id = meta_idx_to_given_idx[str(doc_id)]
# 			ret += str(query_num) + '\t' + str(doc_id) + '\t' + str(score) + '\n'
# 		avg_p = ev.avg_p(results, query_num, num_results)
# 		total += avg_p
# 		print("Query {} average precision: {}".format(query_num, avg_p))
# 	ret = ret.strip('\n')
# 	query_file.close()
# writting_file = open('General_domain_results.txt','w')
# writting_file.write(ret)
# writting_file.close()

# ret = ''
# num_results = 100
# best_score = 0
# best_param = 0 

# for i in np.arange(10.5, 11.5, 0.1):
# 	print("===" + str(i) + "===")
# 	ranker = PL2(i)
# 	# ranker = metapy.index.DirichletPrior(mu=100)
# 	total = 0.0
# 	with open('train_input.txt') as query_file:
# 		for query_num, line in enumerate(query_file):
# 			query = metapy.index.Document()
# 			query.content(line.strip())
# 			results = ranker.score(idx, query, num_results)
# 			avg_p = ev.avg_p(results, query_num, num_results)
# 			total += avg_p
# 		print(total)
# 		if total > best_score:
# 			best_score = total
# 			best_param = i

# print(best_score, best_param)


#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvHYBRIDvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
meta_idx_to_given_idx = None
with open('meta_idx_to_given_idx.json', 'r') as fp:
	meta_idx_to_given_idx = json.load(fp)

bm25 = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500.0)
dirich = metapy.index.DirichletPrior(mu=400)
jm = metapy.index.JelinekMercer()
pl2 = PL2(11)
rankers = [bm25, dirich, jm, pl2]
num_results = 200
total = 0.0
temp = []


for ranker in rankers:
	current  = []
	with open('test_general.txt') as query_file:
		for query_num, line in enumerate(query_file):
			query = metapy.index.Document()
			query.content(line.strip())
			ret = ranker.score(idx, query, num_results)
			current += ret,
		temp += current,
ret = ''
with open('test_general.txt') as query_file:
	for query_num, line in enumerate(query_file):
		dic = collections.defaultdict(list)
		results = []
		for i in range(4):
			for idx, score in temp[i][query_num]:
				dic[idx].append(score)
		for x in sorted(dic, key=lambda x: (len(dic[x]), sum(dic[x])/len(dic[x])), reverse=True)[:100]:
			results.append((x, sum(dic[x])/len(dic[x])))

		for doc_id, score in results:
			doc_id = meta_idx_to_given_idx[str(doc_id)]
			ret += str(query_num) + '\t' + str(doc_id) + '\t' + str(score) + '\n'
	ret = ret.strip('\n')
	query_file.close()

writting_file = open('General_domain_results.txt','w')
writting_file.write(ret)
writting_file.close()
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^HYBRID^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ret = ''
# total = 0.0
# num_results = 100
# result = []
# with open('train_input.txt') as query_file:
# 	for query_num, line in enumerate(query_file):
# 		query.content(line.strip())
# 		results = ranker.score(idx, query, num_results)
# 		# print(results)

# 		avg_p = ev.avg_p(results, query_num, num_results)
# 		total += avg_p
# 		print("Query {} average precision: {}".format(query_num, avg_p))
# 	ret = ret.strip('\n')
# 	query_file.close()

# print("Total average precision: {}".format(total))

"""
# Examine number of documents
idx.num_docs()
# Number of unique terms in the dataset
idx.unique_terms()
# The average document length
idx.avg_doc_length()
# The total number of terms
idx.total_corpus_terms()
"""



# line_number = 0
# dic = collections.defaultdict(lambda: 0)
# with open('./test/test.dat', 'r') as f:
# 	line  = f.readline()
# 	dic[line.split("   ")[0].split("<")[0]] = line_number
# 	while line:
# 		line_number += 1
# 		line = f.readline()
# 		if not line: break
# 		dic[line.split("   ")[0].split("<")[0]] = line_number

# result = []
# with open('./general/train_qrel.txt') as query_file:
# 	for line in query_file:
# 		queryid, docid, relev = map(str, line.strip().split(" "))
# 		result.append(queryid + " " + str(dic[docid]) + " " + relev)
# with open('train_qrel_line.txt', 'w+') as fp:
# 	for item in result:
# 		fp.write("%s\n" % item)

# result  = []
# with open('./academic/test_queries.json') as json_file:
# 	line  = json_file.readline()
# 	result.append(line.split("\"")[-2])
# 	while line:
# 		line  = json_file.readline()
# 		if not line:
# 			break
# 		result.append(line.split("\"")[-2])
	
# with open('test_academic.txt', 'w+') as f:
# 	for item in result:
# 		f.write("%s\n" % item)
