## CS510 Search Engine Competition:

#### Members:
COMPETITION_ALIAS:  Don't Worry
Team Members:
* Rongzi Wang (rwang67)
* Beichen Zhang (bzhang64)
* Qiyang Chen (qiyangc2)
#### Code stucture:
- Code is divided into two main folders: Academic & General


### Method 1: BM25/Dirchlet/JM/PL2
* To generate Academic_domain_results.txt: 
```bash
cd Academic
```
* To generate General_domain_results.txt: 
```bash
cd General
```
- modify the corresponding ranker in test_metapy.py file, then run the following:
```python
python test_metapy.py
```

### Method 2: Majority Voting
- We tune the parameteres for four different rankers (BM25, Dirichlet Smoothing, JM, and PL2)
- To generate the output file for general domain or academic domain dataset,  you have to go into the code to change the input file path
* To generate Academic_domain_results.txt: 
```bash
cd Academic
```
* To generate General_domain_results.txt: 
```bash
cd General
```
```python
python majority_voting.py
```

### Method 3: Supervised Model
* To generate Academic_domain_results.txt: 
```bash
cd Academic/Supervised_Academic
```
* To generate General_domain_results.txt: 
```bash
cd General/Supervised_General
```
- To preprocess the training data, for each training data point (queryid, doc_id, qrel), we use 4 rankers to each generate a score. Each data point is converted into a feature vector of dimension of 4. To run the preprocess step:
 
```python
python supervised.py -p
```
- The processed feature vector is stored as a pickle file called feature_vector.pckl.
- To train a logistic regression model:
```python 
python regresion.py
```
- Given a new query, since the document collection is large, we first use 4 rankers to filter out only the top 5000 documents. For every overlapping document that appear in top 5000 by all 4 rankers, it has a feature vector dimension of 4 ([BM25 score, Dirichlet score, JM score, PL2 score]). We use our trained model to make a prediction and output a vector with dimension of 3 ([prob of label 0, prob of label 1, prob of label 2]). We divide the class into label zero and label non-zero and give more weight to the ones that has more probability for non-zero label. To generate testing prediction:
```python
python supervised.py -t
```