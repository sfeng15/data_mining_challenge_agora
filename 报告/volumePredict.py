import numpy as np
from sklearn import ensemble
import pandas as pd


# use data from past 5 days to predict the volume in 7 days, use GBDT regression model
data = pd.read_csv('usage_sample_2016_8_2-2016_9_1.csv')

res = np.zeros((360, 7))
# use new model to fit each new user,use last 7 chunks to do forecasting
for k in range(360):

	x = [[data.iloc[k][j+i] for i in xrange(5)] for j in xrange(1,20)]
	y = [data.iloc[k][i] for i in xrange(13,32)]
	x_test = [[data.iloc[k][j+i] for i in xrange(5)] for j in xrange(21,28)]

	params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
	          'learning_rate': 0.01, 'loss': 'ls'}
	model = ensemble.GradientBoostingRegressor(**params)
	tmpRes = model.fit(x, y)
	
	est = tmpRes.predict(x_test)
	res[k] = np.array(est)
print res