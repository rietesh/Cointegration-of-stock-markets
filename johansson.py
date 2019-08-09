import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

df_IND = pd.read_csv("final-US.csv")
df_IND = df_IND.open
df_US = pd.read_csv("final-HK.csv")
df_US = df_US.open

df = pd.DataFrame({'x':df_IND,"y":df_US})
df = df.fillna(0)

result = coint_johansen(df,0,1)
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
