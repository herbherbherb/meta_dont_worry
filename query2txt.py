import json

data = []
with open('test_queries.json') as f:
    for line in f:
        doc=json.loads(line)
        data.append(doc["query"])
data = "\n".join(data)

with open("test_queries.txt", "w", encoding="utf8") as file:
    file.write(data)



data = []
with open('train_queries.json') as f1:
    for line in f1:
        doc=json.loads(line)
        data.append(doc["query"])
data = "\n".join(data)

with open("train_queries.txt", "w", encoding="utf8") as file:
    file.write(data)