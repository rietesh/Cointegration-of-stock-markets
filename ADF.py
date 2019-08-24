import pandas as pd
import numpy as np
from pandas import Series
from statsmodels.tsa.stattools import adfuller

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

series = pd.read_csv('final-HK.csv')
X = series.open
# X = np.log(X)
X = difference(X)
result = adfuller(X)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

