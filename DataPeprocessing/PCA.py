import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np 
import matplotlib.pyplot as plt

df_wine = pd.read_csv('D:\Python\WorkSpace\DataPeprocessing\wine.data.txt',header=None)
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
#STEP1
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
#STEP2
cov_mat = np.cov(X_train_std.T)  #calculate covarience matrix of the training set    13x13
# print('covariance matrix:\n',cov_mat)
#STEP3: Decompose the covariance matrix into its eigenvectors and eigenvalues
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print('eigen_vals :\n',eigen_vals)    #13 values
# print('eigen_vecs :\n',eigen_vecs)    #13x13

# tot = sum(eigen_vals)
# var_exp = [(i/tot) for i in sorted(eigen_vals, reverse = True)]
# cum_var_exp = np.cumsum(var_exp)
# plt.bar(range(1,14), var_exp, alpha=0.5, align='center',label='individual explained variance') 
# plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.show()

eigen_pairs =[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
# print('eigen_pairs:\n',eigen_pairs)
eigen_pairs.sort(reverse=True)
# print('eigen_pairs soft:\n',eigen_pairs)
#STEP 4,5: Select 2 eigen pairs on the top of the orded list and build projection matrix
w= np.hstack((eigen_pairs[0][1][:, np.newaxis],eigen_pairs[1][1][:, np.newaxis]))
# print('Matrix W:\n', w)

#STEP 6: Transform d dimensional datas to k dimensional datas
X_train_pca = X_train_std.dot(w)
# color = ['r','b','g']
# markers = ['s','x','o']
# for l,c,m in zip(np.unique(y_train), color, markers):
#     plt.scatter(X_train_pca[y_train==l,0],X_train_pca[y_train==l,1], c =c, label=l, marker=m)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.show()



######################## PCA in scikit learn###################
# Although the verbose approach in the source code above, actually,
# PCA have been implemented in scikit learn.... lets see: 
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)   # choose k = 2
