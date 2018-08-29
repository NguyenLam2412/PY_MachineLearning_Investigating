import pandas as pd 
from io import StringIO 
csv_data = ''' A,B,C,D
...1.0,2.0,3.0,4.0
...5.0,6.0,,8.0
...10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
print('df\n',df)
print('df value:\n',df.values)
numberOfNullData = df.isnull().sum()
print(numberOfNullData)

removeSampleWithNull = df.dropna()
print('removeSampleWithNull:\n', removeSampleWithNull)

removeFeatureWithNull = df.dropna(1)
print('removeFeatureWithNull:\n', removeFeatureWithNull)

dropSampleAllNull = df.dropna(how='all')    #drop rows where all collumns are Null
print('dropSampleAllNull:\n', dropSampleAllNull)

dropSampleWith4NullValue = df.dropna(thresh=4)    #drop rows where at least have 4 non NaN value
print('dropSampleWith4NullValue:\n', dropSampleWith4NullValue)

dropSampleWhereNullOnFeatureC = df.dropna(subset=['C'])    #drop rows where all collumns are NaN
print('dropSampleWhereNullOnFeatureC:\n', dropSampleWhereNullOnFeatureC)