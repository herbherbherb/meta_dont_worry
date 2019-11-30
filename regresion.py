import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

def main():
	feature = pickle.load(open('feature_vector.pckl', 'rb'))
	x = map(lambda x:x[:-1], feature)
	y = map(lambda x:x[-1], feature)
	print(len(x), len(y))
	print(x[:3], y[:3])
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