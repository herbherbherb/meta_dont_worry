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
import collections
import pickle
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from heapq import heappush, heappop


idx = metapy.index.make_inverted_index('config.toml')
query = metapy.index.Document()
bm25 = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500.0)
dirich = metapy.index.DirichletPrior(mu=400)
jm = metapy.index.JelinekMercer()
pl2 = PL2(11)
rankers = [bm25, dirich, jm, pl2]
# rankers = [bm25]
mapper = []
# fidx = metapy.index.make_forward_index('./config.toml')
# dset = metapy.learn.Dataset(fidx)

ev = metapy.index.IREval('config.toml')

meta_idx_to_given_idx = None
with open('meta_idx_to_given_idx.json', 'r') as fp:
	meta_idx_to_given_idx = json.load(fp)


#=============================================================================
num_results = 100000
for ranker in rankers:
	dic = collections.defaultdict(lambda : 0)
	with open('train_input.txt') as query_file:
		for query_num, line in enumerate(query_file):
			query = metapy.index.Document()
			query.content(line.strip())
			results = ranker.score(idx, query, num_results)
			for doc_id, score in results:
				actual_id = meta_idx_to_given_idx.get(str(doc_id), "")
				dic[(str(query_num), str(actual_id))] = score
	mapper.append(dic)

feature_vector = []
with open('./general/train_qrel.txt') as query_file:
	for line in query_file:
		queryid, doc_id, label = map(str, line.strip().split(" "))
		cur = []
		for dic in mapper:
			cur.append(dic[(queryid, doc_id)])
		cur.append(int(label))
		feature_vector += cur,
		
pickle.dump(feature_vector, open('feature_vector.pckl', 'wb'))

# with open('feature_vector.txt', 'w+') as f:
# 	for item in feature_vector:
# 		f.write("%s\n" % item)
#=============================================================================
# logisticRegr = load('logisticRegr.joblib') 


# num_results = 5000
# with open('test_general.txt') as query_file:
# 	for query_num, line in enumerate(query_file):
# 		mapper.append(collections.defaultdict(list))

# for ranker in rankers:
# 	with open('test_general.txt') as query_file:
# 		for query_num, line in enumerate(query_file):
# 			query = metapy.index.Document()
# 			query.content(line.strip())
# 			results = ranker.score(idx, query, num_results)
# 			for doc_id, score in results:
# 				actual_id = meta_idx_to_given_idx.get(str(doc_id), "")
# 				mapper[query_num][actual_id].append(score)

# ret = ''
# with open('test_general.txt') as query_file:
# 	for query_num, line in enumerate(query_file):
# 		dic = mapper[query_num]
# 		result = []
# 		heap = []
# 		for key in dic:
# 			val = dic[key]
# 			if len(val) == 4:
# 				res = logisticRegr.predict_proba([val])[0]
# 				label = np.argmax(res)
# 				prob = max(res) * (label+1) * 10
# 				heappush(heap, (-prob, key))
# 		for i in range(100):
# 			prob, idx = heappop(heap)
# 			prob = -prob
# 			ret += str(query_num) + '\t' + str(doc_id) + '\t' + str(prob) + '\n'
# 	ret = ret.strip('\n')
# 	query_file.close()
# writting_file = open('General_domain_results.txt','w')
# writting_file.write(ret)
# writting_file.close()


