from sklearn.preprocessing import Imputer
import pandas as pd 
from io import StringIO 

csv_data = ''' A,B,C,D
 1.0,2.0,3.0,4.0
 5.0,6.0,,8.0
 10.0, 11.0,10.0,'''
df = pd.read_csv(StringIO(csv_data))
print('df:\n', df)

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df)
print('imputed_data mean feature:\n', imputed_data)

imr = Imputer(missing_values='NaN', strategy='mean', axis=1)
imr = imr.fit(df)
imputed_data = imr.transform(df)
print('imputed_data mean sample:\n', imputed_data)

imr = Imputer(missing_values='NaN', strategy='median', axis=1)
imr = imr.fit(df)
imputed_data = imr.transform(df)
print('imputed_data median:\n', imputed_data)

imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
imr = imr.fit(df)
imputed_data = imr.transform(df)
print('imputed_data most frequent:\n', imputed_data)