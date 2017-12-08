from sklearn import cluster, datasets
import pandas as pd
from sklearn import preprocessing
import numpy as np


data = pd.read_csv('test.csv')
m = data.as_matrix()
newM = [m[i][1:] for i in xrange(len(m))]
# print newM[5]

X_normalized = preprocessing.normalize(newM, norm='l2')
X_normalized
# print X_normalized[5]

k_means = cluster.KMeans(n_clusters=2, init='k-means++')
k_means.fit(X_normalized)
print k_means.labels_

np.savetxt('output.csv', X_normalized, delimiter = ",")