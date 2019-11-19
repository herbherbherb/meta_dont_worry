import metapy
import os
import json
import re
import traceback
import xml.etree.ElementTree as ET
import collections

with open("mapping_rev.json",'r') as file:
    d = json.load(file)
#print(d)

idx = metapy.index.make_inverted_index('config.toml')
query = metapy.index.Document()
ranker = metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500)
#ranker = metapy.index.DirichletPrior(mu=1)

print(idx.num_docs())

ev = metapy.index.IREval('config.toml')

num_results = 100
print(idx)
output = []
with open('test_queries.txt') as query_file:
    for query_num, line in enumerate(query_file):
        query.content(line.strip())
        results = ranker.score(idx, query, num_results)
        output_row = [ str(query_num) + '\t' + d[str(id)]+ '\t' + str(score) for (id,score) in results]
        print(output_row)
        output.append("\n".join(output_row))
        #avg_p = ev.avg_p(results, query_num, num_results)
        #print("Query {} average precision: {}".format(query_num, avg_p))
#print(ev.map())

output = "\n".join(output)
with open('academic.txt','w') as f:
    f.write(output)

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
# tree = ET.parse('./general/trainqueries.xml')
# root = tree.getroot()
# dic = collections.OrderedDict()
# for q in root.findall('query'):
# 	cur = q[1].text.split("( ")[1]
# 	dic[q[0].text] = cur[:-2]
# 	result.append(cur[:-2])
# with open('train_query.json', 'w') as fp:
#     json.dump(dic, fp)

# with open('train_input.txt', 'w+') as f:
# 	for item in result:
# 		f.write("%s\n" % item)

# relev_dic = {}
# with open('./general/train_qrel') as query_file:
# 	for line in query_file:
# 		queryid, docid, relev = map(str, line.strip().split(" "))
# 		relev_dic[(docid, queryid)] = relev
# with open('relevancy.json', 'w') as fp:
#     json.dump(relev_dic.items(), fp)

# with open('train_query.json') as json_file:
#     data = json.load(json_file)
