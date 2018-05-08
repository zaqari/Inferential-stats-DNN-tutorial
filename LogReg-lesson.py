import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
import scipy.stats as st
import statsmodels.api as sm

df_1y = pd.read_csv("~/data/FirstYear-post.csv", skipinitialspace=True)
df_Uy = pd.read_csv("~/data/Upperclassmen-post.csv", skipinitialspace=True)


#####
##FUNCTION
#####
def final_data(dfk, cols_toreplace):

	for col in cols_toreplace:
		for it in list(set(dfk[col].values.tolist())):
			a=it
			dfk[col] = dfk[col].replace([a], list(set(dfk[col].values.tolist())).index(a))

	return dfk


#####
##IMPLEMENTATION
#####
lr=LogReg(solver='newton-cg', multi_class='multinomial')

df2=df_1y.replace([np.nan], 'NA')
df3=df_Uy.replace([np.nan], 'NA')

df1y=final_data(df2, list(df2)[1:])
dfUy=final_data(df3, list(df3)[1:])

###
#Q5 RESULTS
###

#1y Students
print('===1y Q5===')
x=df1y[['Q5B', 'Q5C']]
y=df1y['Q5A']
x2=sm.add_constant(x)
est=sm.GLM(y, x2)
est2=est.fit()
print(est2.summary())

#Uy students
print('===Uy Q5===')
x=dfUy[['Q5B', 'Q5C']]
y=dfUy['Q5A']
x2=sm.add_constant(x)
est=sm.GLM(y, x2)
est2=est.fit()
print(est2.summary())


###
#Q6 RESULTS
###

#1y Students
print('===1y Q6===')
x=df1y[['Q6B', 'Q6C']]
y=df1y['Q6A']
x2=sm.add_constant(x)
est=sm.GLM(y, x2)
est2=est.fit()
est2.summary()

#Uy students
print('===Uy Q6===')
x=dfUy[['Q6B', 'Q6C']]
y=dfUy['Q6A']
x2=sm.add_constant(x)
est=sm.GLM(y, x2)
est2=est.fit()
print(est2.summary())

###
#Q7 RESULTS
###

#1y Students
print('===1y Q7===')
x=df1y[['Q7B', 'Q7C']]
y=df1y['Q7A']
x2=sm.add_constant(x)
est=sm.GLM(y, x2)
est2=est.fit()
print(est2.summary())

#Uy students
print('===Uy Q7===')
x=dfUy[['Q7B', 'Q7C']]
y=dfUy['Q7A']
x2=sm.add_constant(x)
est=sm.GLM(y, x2)
est2=est.fit()
print(est2.summary())




#lr.fit(df1y[list(df1y)[3:]], df1y['Q4'])
#y=lr.score(df1y[list(df1y)[3:]], df1y['Q4'])

#print(y)

#lr.fit(dfUy[list(dfUy)[3:]], dfUy['Q4'])
#y=lr.score(dfUy[list(dfUy)[3:]], dfUy['Q4'])

#print(y)

#df.to_csv('/Volumes/HOLOCRON/FRESH-START-DATA/cleaned-data/regression-csvs/FirstYear-post.csv', sep=',', header=True, index=False, encoding='utf-8')

