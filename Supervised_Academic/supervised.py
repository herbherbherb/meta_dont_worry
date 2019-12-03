import metapy
import os
import json
import re
import traceback
from pl2 import load_ranker
from pl2 import PL2
import collections
import pickle
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from heapq import heappush, heappop
import sys

with open("./mapping_rev.json",'r') as file:
    meta_idx_to_given_idx = json.load(file)

idx = metapy.index.make_inverted_index('config.toml')
bm25 = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500.0)
dirich = metapy.index.DirichletPrior(mu=0.05)
jm = metapy.index.JelinekMercer()
pl2 = PL2(1)
rankers = [bm25, dirich, jm, pl2]
ev = metapy.index.IREval('config.toml')
mapper = []

flag = sys.argv[1]
#==============================================================================================
if flag == '-p':
    num_results = 50000
    for ranker in rankers:
        dic = collections.defaultdict(lambda : 0)
        with open('train_queries.txt') as query_file:
            for query_num, line in enumerate(query_file):
                query = metapy.index.Document()
                query.content(line.strip())
                results = ranker.score(idx, query, num_results)
                for doc_id, score in results:
                    actual_id = meta_idx_to_given_idx.get(str(doc_id), "")
                    dic[(str(query_num), str(actual_id))] = score
        mapper.append(dic)
    print("Finish ranking")
    feature_vector = []
    with open('train_queries_qrel.txt') as query_file:
        for line in query_file:
            queryid, doc_id, label = map(str, line.strip().split(" "))
            cur = []
            for dic in mapper:
                cur.append(dic[(queryid, doc_id)])
            cur.append(int(label))
            feature_vector += cur,
            
    pickle.dump(feature_vector, open('feature_vector.pckl', 'wb'))

    with open('feature_vector.txt', 'w+') as f:
        for item in feature_vector:
            f.write("%s\n" % item)
#==============================================================================================
elif flag == '-t':
    print("Start training")
    logisticRegr = load('logisticRegr.joblib')
    num_results = 200
    with open('test_queries.txt') as query_file:
        for query_num, line in enumerate(query_file):
            mapper.append(collections.defaultdict(list))

    for ranker in rankers:
        with open('test_queries.txt') as query_file:
            for query_num, line in enumerate(query_file):
                query = metapy.index.Document()
                query.content(line.strip())
                results = ranker.score(idx, query, num_results)
                for doc_id, score in results:
                    actual_id = meta_idx_to_given_idx.get(str(doc_id), "")
                    mapper[query_num][actual_id].append(score)

    ret = ''
    with open('test_queries.txt') as query_file:
        for query_num, line in enumerate(query_file):
            dic = mapper[query_num]
            result = []
            heap = []
            for key in dic:
                val = dic[key]
                if len(val) == 4:
                    res = logisticRegr.predict_proba([val])[0]
                    if sum(res[1:]) > res[0]:
                        prob = 1 + sum(res[1:])
                    else:
                        prob = res[0]
                    heappush(heap, (-prob, key))
            print(len(heap))
            for i in range(min(len(heap), 100)):
                prob, idx = heappop(heap)
                prob = -prob
                ret += str(query_num) + '\t' + str(idx) + '\t' + str(prob) + '\n'
        ret = ret.strip('\n')
        query_file.close()
    writting_file = open('Academic_domain_results.txt','w')
    writting_file.write(ret)
    writting_file.close()
#==============================================================================================
else:
    raise ValueError("\n Preprocess training data: python supervised.py -p \n Predict on testing data: python supervised.py -t")
