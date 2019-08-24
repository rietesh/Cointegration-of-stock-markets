import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

df_1 = pd.read_csv("final-US.csv")
df_2 = pd.read_csv("final-IND.csv")

df_1 = np.log(df_1.open)
df_2 = np.log(df_2.open)
df = pd.DataFrame({'x':df_1,"y":df_2})
df = df.fillna(0)

result = coint_johansen(df,0,1)

print('---------------------------------------------------')
print('IND-US')
print ('--------------------------------------------------')
print ('--> Trace Statistics')
print ('variable statistic Crit-90% Crit-95%  Crit-99%')
for i in range(len(result.lr1)):
    print ('r =', i, '\t', round(result.lr1[i], 4), result.cvt[i, 0], result.cvt[i, 1], result.cvt[i, 2])
print ('--------------------------------------------------')
print ('--> Eigen Statistics')
print ('variable statistic Crit-90% Crit-95%  Crit-99%')
for i in range(len(result.lr2)):
    print ('r =', i, '\t', round(result.lr2[i], 4), result.cvm[i, 0], result.cvm[i, 1], result.cvm[i, 2])
print ('--------------------------------------------------')
print ('eigenvectors:\n', result.evec)
print ('--------------------------------------------------')
print ('eigenvalues:\n', result.eig)
print ('--------------------------------------------------')
