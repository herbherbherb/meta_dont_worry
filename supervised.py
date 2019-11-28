import metapy
import numpy as np
import os
import json
import re
import traceback
import xml.etree.ElementTree as ET
import collections
# from search_eval import load_ranker
# from search_eval import PL2Ranker

idx = metapy.index.make_inverted_index('config.toml')
query = metapy.index.Document()
ranker = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500.0)
# ranker = metapy.index.DirichletPrior(mu=100)
# ranker = metapy.index.JelinekMercer()
# ranker = PL2Ranker(1)
# fidx = metapy.index.make_forward_index('./config.toml')
# dset = metapy.learn.Dataset(fidx)
ev = metapy.index.IREval('config.toml')

with open('train_query.json', 'r') as fp:
	train_query = json.load(fp)
print(len(train_query))