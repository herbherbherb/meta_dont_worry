import metapy
import numpy as np
import os
import json
import re
import traceback
import collections
from pl2 import load_ranker
from pl2 import PL2

idx = metapy.index.make_inverted_index('config.toml')
ev = metapy.index.IREval('config.toml')

with open("mapping_rev.json",'r') as file:
    meta_idx_to_given_idx = json.load(file)

bm25 = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500.0)
dirich = metapy.index.DirichletPrior(mu=0.05)
jm = metapy.index.JelinekMercer()
pl2 = PL2(1)
rankers = [bm25, dirich, jm, pl2]
ranker = rankers[0]
num_results = 100
output = []
with open('test_queries.txt') as query_file:
    for query_num, line in enumerate(query_file):
    	query = metapy.index.Document()
        query.content(line.strip())
        results = ranker.score(idx, query, num_results)
        output_row = [ str(query_num) + '\t' + meta_idx_to_given_idx.get(str(iidx), "")+ '\t' + str(score) for (iidx, score) in results]
        output.append("\n".join(output_row))

output = "\n".join(output)
with open('Academic_domain_results.txt','w') as f:
    f.write(output)
