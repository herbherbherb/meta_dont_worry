import metapy
import numpy as np
import os
import json
import re
import traceback
import xml.etree.ElementTree as ET
import collections
from search_eval import load_ranker
from search_eval import PL2Ranker

idx = metapy.index.make_inverted_index('config.toml')
query = metapy.index.Document()
ranker = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500.0)
# ranker = metapy.index.DirichletPrior(mu=100)
# ranker = metapy.index.JelinekMercer()
# ranker = PL2Ranker(1)
# fidx = metapy.index.make_forward_index('./config.toml')
# dset = metapy.learn.Dataset(fidx)
ev = metapy.index.IREval('config.toml')

# line_number = 0
# dic = collections.defaultdict(lambda: 0)
# with open('./test/test.dat', 'r') as f:
# 	line  = f.readline()
# 	dic[line_number] = line.split("   ")[0].split("<")[0]
# 	while line:
# 		line_number += 1
# 		line = f.readline()
# 		if not line: break
# 		dic[line_number] = line.split("   ")[0].split("<")[0]
# with open('line_to_general_file.json', 'w+') as fp:
# 	json.dump(dic, fp)

# line_to_general_file = None
# with open('line_to_general_file.json', 'r') as fp:
# 	line_to_general_file = json.load(fp)
# ret = ''
# total = 0.0
# num_results = 100
# with open('train_input.txt') as query_file:
# 	for query_num, line in enumerate(query_file):
# 		query.content(line.strip())
# 		results = ranker.score(idx, query, num_results)
# 		# print(results)
# 		for doc_id, score in results:
# 			doc_id = line_to_general_file[str(doc_id)]
# 			ret += str(query_num) + '\t' + str(doc_id) + '\t' + str(score) + '\n'
# 		avg_p = ev.avg_p(results, query_num, num_results)
# 		total += avg_p
# 		print("Query {} average precision: {}".format(query_num, avg_p))
# 	ret = ret.strip('\n')
# 	query_file.close()
# writting_file = open('General_domain_results.txt','w')
# writting_file.write(ret)
# writting_file.close()

line_to_general_file = None
with open('line_to_general_file.json', 'r') as fp:
	line_to_general_file = json.load(fp)
ret = ''
total = 0.0
num_results = 100
with open('train_input.txt') as query_file:
	for query_num, line in enumerate(query_file):
		query.content(line.strip())
		results = ranker.score(idx, query, num_results)
		# print(results)
		modified_results = []
		for doc_id, score in results:
			doc_id = line_to_general_file[str(doc_id)]
			print(type(doc_id))
			ret += str(query_num) + '\t' + str(doc_id) + '\t' + str(score) + '\n'
			modified_results.append((doc_id, score))
		avg_p = ev.avg_p(modified_results, query_num, num_results)
		total += avg_p
		print("Query {} average precision: {}".format(query_num, avg_p))
	ret = ret.strip('\n')
	query_file.close()

print("Total average precision: {}".format(total))

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
# idx = None
# idx = metapy.index.make_inverted_index('./config.toml')
# print(idx.num_docs())
# print('Finished indexing input')

# query = metapy.index.Document()
# print(idx.unique_terms())
# ranker = metapy.index.OkapiBM25(k1=1.2,b=0.75,k3=500)
# ev = metapy.index.IREval('./config.toml')

# fidx = metapy.index.make_forward_index('./config.toml')
# dset = metapy.learn.Dataset(fidx) #527747
# idx = metapy.index.make_inverted_index('./config.toml')
# metapy.learn.tfidf_transform(dset, idx, metapy.index.OkapiBM25()) # or any other ranker

# print(dset[0]. ) # (FeatureVector) that contains all of the non-zero feature counts
# print(dset[0].id) # doc id



# ranker = metapy.index.OkapiBM25()
# query = metapy.index.Document()
# query.content('compost pile')


# result = []
# tree = ET.parse('./general/testqueries.xml')
# root = tree.getroot()
# dic = collections.OrderedDict()
# for q in root.findall('query'):
# 	cur = q[1].text.split("( ")[1]
# 	dic[q[0].text] = cur[:-2]
# 	result.append(cur[:-2])
# with open('test_query.json', 'w') as fp:
# 	json.dump(dic, fp)

# with open('input.txt', 'w') as f:
# 	for item in result:
# 		f.write("%s\n" % item)

# result = []
# tree = ET.parse('./general/testqueries.xml')
# root = tree.getroot()
# dic = collections.OrderedDict()
# for q in root.findall('query'):
# 	cur = q[1].text.split("( ")[1]
# 	dic[q[0].text] = cur[:-2]
# 	result.append(cur[:-2])
# with open('train_query.json', 'w') as fp:
#     json.dump(dic, fp)

# with open('test_general.txt', 'w+') as f:
# 	for item in result:
# 		f.write("%s\n" % item)


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
