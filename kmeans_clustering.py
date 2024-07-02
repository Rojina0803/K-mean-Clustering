#!UNSUPERVISED LEARNING : outputs is extreacted based on the inputs?dataset we provided. // the "y" parametre is not provided.
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv('income.csv')
# print(df.head())

#! EDA 
# plt.scatter(df['Age'], df['Income($)'])
# plt.xlabel('Age')
# plt.ylabel('Income')
# plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters= 3)
predicting = km.fit_predict(df[['Age', 'Income($)']])
df['cluster'] = predicting
# print(df.head())
# print(predicting)

df1 = df[df['cluster']== 0]
df2 = df[df['cluster']==1]
df3 = df[df['cluster']==2]

# print(df1)
# print(df2)
# print(df3)

# plt.scatter(df1['Age'], df1['Income($)'], color = 'green')
# plt.scatter(df2['Age'], df2['Income($)'], color = 'red')
# plt.scatter(df3['Age'], df3['Income($)'], color = 'blue')
# plt.show()

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()

scalar.fit (df[["Income($)"]])
df['Income($)'] = scalar.transform(df[['Income($)']])


scalar.fit(df[['Age']])
df['Age'] = scalar.transform(df[['Age']])

km1 = KMeans(n_clusters= 3)
y_pred= km1.fit_predict(df[['Age', 'Income($)']])

df['cluster'] = y_pred
# print(df.head())
# print(predicting)

df1 = df[df['cluster']== 0]
df2 = df[df['cluster']==1]
df3 = df[df['cluster']==2]

# plt.scatter(df1['Age'], df1['Income($)'], color = 'green')
# plt.scatter(df2['Age'], df2['Income($)'], color = 'red')
# plt.scatter(df3['Age'], df3['Income($)'], color = 'blue')
# plt.show()

sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

plt.plot(k_rng,sse)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.show()