import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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
	logisticRegr = load('logisticRegr.joblib') 
	predictions = logisticRegr.predict(x)
	print(collections.Counter(predictions))
	print(collections.Counter(y))
	print(np.mean(predictions == y))



if __name__ == "__main__":
	main()


# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# # y = 1 * x_0 + 2 * x_1 + 3
# y = np.dot(X, np.array([1, 2])) + 3
# reg = LinearRegression().fit(X, y)
# reg.score(X, y)

# reg.coef_

# reg.intercept_ 

# reg.predict(np.array([[3, 5]]))