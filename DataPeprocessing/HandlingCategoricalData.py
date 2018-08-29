import pandas as pd 
import numpy as np

from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame([
 ['green', 'M', 10.1, 'class1'],
 ['red', 'L', 13.5, 'class2'],
 ['blue', 'XL', 15.3, 'class1']])

df.columns =['color', 'size', 'price','classlabel']
print('dataset:\n', df)

size_mapping = {
        'XL':3,
        'L':2,
        'M':1}

df['size'] = df['size'].map(size_mapping)
print('dataset_mapping:\n', df)

inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
print('dataset_RevertMapping:\n', df)

class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
print('Class mapping:\n', class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print('dataset_classMapping:\n', df)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print('dataset_RevertClassMapping:\n', df)

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
x = [1,1,0,1]
y_inverse = class_le.inverse_transform(x)
print(y_inverse)