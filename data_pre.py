import numpy as np
import re
import pickle
## read raw data ##
label = []
feature = []
cash_feature = []
feature = pickle.load(open('feature_vector.pckl', 'rb'))
# exit()
x = map(lambda x:x[:-1], feature)
y = map(lambda x:x[-1], feature)
x = list(x)
x = np.array(x)
y= np.array(list(y))
np.save('train_data.npy',x)
np.save('train_label.npy',y)
print('Data saving is done !')