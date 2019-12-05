import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
from joblib import dump, load
import collections

def main():
	feature = pickle.load(open('feature_vector.pckl', 'rb'))
	x = map(lambda x:x[:-1], feature)
	y = map(lambda x:x[-1], feature)
	logisticRegr = LogisticRegression()
	logisticRegr.fit(x, y)
	dump(logisticRegr, 'logisticRegr.joblib')
	predictions = logisticRegr.predict(x)
	print(collections.Counter(predictions))
	print(collections.Counter(y))
	print(np.mean(predictions == y))

if __name__ == "__main__":
	main()
