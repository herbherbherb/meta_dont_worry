import json
import csv

with open("mapping.json",'r') as file:
    d = json.load(file)

result = []
with open("train_queries_qrel.txt",'r') as old:
    read = csv.reader(old)
    for row in read:
        newrow = row[0].split(" ")
        if newrow[1] not in d:
            continue
        newrow[1] = str(d[newrow[1]])
        newrow = " ".join(newrow)
        result.append(newrow)

result = "\n".join(result)
with open("rel.txt",'w') as new:
    new.write(result)
