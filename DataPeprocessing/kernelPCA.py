from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np 

def rbf_kernel_pca(X, gamma, n_components):
    # Calcilate pairwise squared euclidean distances
    sq_dists = pdist(X, 'sqeuclidean')
    # convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)
    # Compute the symetric kernel matrix
    K = exp(-gamma*mat_sq_dists)
    #center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N))/N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    #obtaining eigenpairs from the centerd kernel matrix
    eigvals, eigvecs = eigh(K)
    # print('eigvals:\n', eigvals)
    #collect the top k eigenvectors 
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1, n_components+1)))
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]
    # print('lambdas:\n', lambdas)
    return alphas, lambdas

def project_x(x_new, X, gamma, alphas, lambdas):
        pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
        k = np.exp(-gamma*pair_dist)
        return k.dot(alphas/lambdas)
X,y = make_moons(n_samples=100, random_state=123)

alphas,lambdas = rbf_kernel_pca(X, gamma= 15, n_components=1)

# fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7,3))
# ax[0].scatter(X_kpca[y==0,0], X_kpca[y==0,1], color = 'red', marker = '^', alpha = 0.5)
# ax[0].scatter(X_kpca[y==1,0], X_kpca[y==1,1], color = 'blue', marker = 'o', alpha = 0.5)
# ax[1].scatter(X_kpca[y==0,0], np.zeros((50,1))+0.02, color = 'red', marker = '^', alpha = 0.5)
# ax[1].scatter(X_kpca[y==1,0], np.zeros((50,1))+0.02, color = 'blue', marker = 'o', alpha = 0.5)
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# plt.show()
x_new = X[25]
x_pro = project_x(x_new,X, gamma = 15,alphas = alphas, lambdas = lambdas)
print(x_pro)