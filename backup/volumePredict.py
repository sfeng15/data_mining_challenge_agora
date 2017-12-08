import numpy as np
from sklearn import ensemble
import pandas as pd



data = pd.read_csv('usage_sample_2016_8_2-2016_9_1.csv')

res = np.zeros((360, 7))
for k in range(360):
	x = [[data.iloc[k][j+i] for i in xrange(5)] for j in xrange(1,20)]
	y = [data.iloc[k][i] for i in xrange(13,32)]
	# print 'x', len(x)
	# print 'y', len(y)
	# print 'x', x
	# print 
	# print 'y', y
	# print 

	x_test = [[data.iloc[k][j+i] for i in xrange(5)] for j in xrange(21,28)]
	# print 'x_test',x_test

	params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
	          'learning_rate': 0.01, 'loss': 'ls'}
	model = ensemble.GradientBoostingRegressor(**params)
	tmpRes = model.fit(x, y)
	
	# validate prediction model works well
	# est = tmpRes.predict([[data.iloc[k][j+i] for i in xrange(5)] for j in xrange(1,20)])
	est = tmpRes.predict(x_test)
	# print 
	# print 'est',est
	
	res[k] = np.array(est)