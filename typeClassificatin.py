from sklearn import cluster, datasets
import pandas as pd
from sklearn import preprocessing
import numpy as np

# clean the data and normalize the data to make it comparable among different datapoint
# use kmeans clustering algorithm to cluster into 2 classes, use k-means++
# to choose initial points

# preprocess data
data = pd.read_csv('test.csv')
m = data.as_matrix()
newM = [m[i][1:] for i in xrange(len(m))]
X_normalized = preprocessing.normalize(newM, norm='l2')


# perform clustering
k_means = cluster.KMeans(n_clusters=2, init='k-means++')
k_means.fit(X_normalized)

# output results
print k_means.labels_
np.savetxt('output.csv', X_normalized, delimiter = ",")