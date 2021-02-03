import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from sklearn.datasets import load_iris
from pandas.api.types import is_numeric_dtype

data = pd.read_csv('C:\\Users\\Goran\\Desktop\\iris_data.csv',header=None)
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
#data.head()
#print(data)
#np.random.seed(0)
#df = pd.DataFrame({"a": np.random.random_integers(1, high=100, size=100)})
#ranges = [0,10,20,30,40,50,60,70,80,90,100]
#df.groupby(pd.cut(df.a, ranges)).count()

print(data['class'].value_counts())#normalize=True))

data1 = data.loc[data['class'] == 'Iris-versicolor']
data1.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

data2 = data.loc[data['class'] == 'Iris-setosa']
data2.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

data3 = data.loc[data['class'] == 'Iris-virginica']
data3.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

for col in data[['sepal length', 'sepal width']]:
    if is_numeric_dtype(data[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % data[col].mean())
        print('\t Median = %.2f' % data[col].median())
        print('\t Standard deviation = %.2f' % data[col].std())
        print('\t Minimum = %.2f' % data[col].min())
        print('\t Maximum = %.2f' % data[col].max())
print('\n')
print(data.groupby(pd.cut(data['sepal length'], np.arange(0,10))).size())
print('\nin %')
print(data.groupby(pd.cut(data['sepal length'], np.arange(0,10))).size().transform(lambda x: x*100/sum(x)))
print('\n')
print(data.groupby(pd.cut(data['sepal width'], np.arange(0,10))).size())
print('\nin %')
print(data.groupby(pd.cut(data['sepal width'], np.arange(0,10))).size().transform(lambda x: x*100/sum(x)))

print('Iris-versicolor\n')
for col in data1[['sepal length', 'sepal width']]: #.columns:
    if is_numeric_dtype(data1[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % data1[col].mean())
        print('\t Median = %.2f' % data1[col].median())
        print('\t Standard deviation = %.2f' % data1[col].std())
        print('\t Minimum = %.2f' % data1[col].min())
        print('\t Maximum = %.2f' % data1[col].max())
print('\n')
print(data1.groupby(pd.cut(data1['sepal length'], np.arange(0,10))).size())
print('\nin %')
print(data1.groupby(pd.cut(data1['sepal length'], np.arange(0,10))).size().transform(lambda x: x*100/sum(x)))
print('\n')
print(data1.groupby(pd.cut(data1['sepal width'], np.arange(0,10))).size())
print('\nin %')
print(data1.groupby(pd.cut(data1['sepal width'], np.arange(0,10))).size().transform(lambda x: x*100/sum(x)))

print('Iris-setosa\n')
for col in data2[['sepal length', 'sepal width']]: 
    if is_numeric_dtype(data2[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % data2[col].mean())
        print('\t Median = %.2f' % data2[col].median())
        print('\t Standard deviation = %.2f' % data2[col].std())
        print('\t Minimum = %.2f' % data2[col].min())
        print('\t Maximum = %.2f' % data2[col].max())
print('\n')
print(data2.groupby(pd.cut(data2['sepal length'], np.arange(0,10))).size())
print('\nin %')
print(data2.groupby(pd.cut(data2['sepal length'], np.arange(0,10))).size().transform(lambda x: x*100/sum(x)))
print('\n')
print(data2.groupby(pd.cut(data2['sepal width'], np.arange(0,10))).size())
print('\nin %')
print(data2.groupby(pd.cut(data2['sepal width'], np.arange(0,10))).size().transform(lambda x: x*100/sum(x)))

print('Iris-virginica\n')
for col in data3[['sepal length', 'sepal width']]: 
    if is_numeric_dtype(data3[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % data3[col].mean())
        print('\t Median = %.2f' % data3[col].median())
        print('\t Standard deviation = %.2f' % data3[col].std())
        print('\t Minimum = %.2f' % data3[col].min())
        print('\t Maximum = %.2f' % data3[col].max())
print('\n')
print(data3.groupby(pd.cut(data3['sepal length'], np.arange(0,10))).size())
print('\nin %')
print(data3.groupby(pd.cut(data3['sepal length'], np.arange(0,10))).size().transform(lambda x: x*100/sum(x)))
print('\n')
print(data3.groupby(pd.cut(data3['sepal width'], np.arange(0,10))).size())
print('\nin %')
print(data3.groupby(pd.cut(data3['sepal width'], np.arange(0,10))).size().transform(lambda x: x*100/sum(x)))