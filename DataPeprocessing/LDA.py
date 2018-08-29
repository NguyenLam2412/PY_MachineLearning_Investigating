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
#STEP2: calculate the mean vector for each class
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
#STEP3: Calculate the within class scatter matrix Sw
# d = 13
# S_W = np.zeros((d,d))
# for label, mv in zip(range(1,4), mean_vecs):
#     class_scatter = np.zeros((d, d))
#     for row in X_train[y_train==label]:
#         row, mv = row.reshape(d,1),mv.reshape(d,1)
#         class_scatter += (row-mv).dot((row-mv).T)
#     S_W += class_scatter
# print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
d = 13 # number of features
S_W = np.zeros((d, d))
for label,mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
# print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

#STEP 3: Calculate between class scatter matrix Sb
mean_overall = np.mean(X_train_std,axis=0)
d=13
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train==i+1,:].shape[0] # number of sample in class 1 in X_train
    mean_vec = mean_vec.reshape(d,1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('Between-class scatter matrix: %sx%s'% (S_B.shape[0], S_B.shape[1]))

#STEP 4: compute eigenvectors and eigenvalues:
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key= lambda k:k[0] , reverse=True)

#STEP 5: choose 2 most discriminative eigenvector collums to create the transfomation matrix w:
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,eigen_pairs[1][1][:, np.newaxis].real))

#STEP 6: project sample onto the new feature space using transformation matrix W
X_train_lda = X_train.dot(w)


# Just like PCA, those verbose steps above are for us to understanding inside LDA
# scikit learn have been implemented LDA 
#from sklear.lda import LDA
# lda = LDA(n_components=2)
# X_train_lda = lda.fit_transform(X_train_std,y_train)
#lr = LogisticRegression()
#lr = lr.fit(X_train_lda,y_train)