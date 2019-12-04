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
ev = metapy.index.IREval('config.toml')

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
