import pandas as pd
import numpy as np
from pandas import Series
from statsmodels.tsa.stattools import adfuller

series = pd.read_csv('final-HK.csv')
X = series.open

X = np.log(X)
result = adfuller(X)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
