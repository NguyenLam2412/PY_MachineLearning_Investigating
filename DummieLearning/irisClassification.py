import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np 
import PerceptronAlgorithm
import AdalineAlgorithm
import AdalineSGDAlgorithm

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)

df = pd.read_csv('D:/Python/WorkSpace/DummieLearning/iris.data.txt')
# print(df.tail())
y_train = df.iloc[0:100, 4].values
y_train = np.where(y_train == 'Iris-setosa',-1,1)
# print(y)
X_train = df.iloc[0:100, [0,2]].values
# print(X)
# plt.scatter(X[:50,0], X[:50,1],color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
# plt.xlabel('sepal length')
# plt.ylabel('petal length')
# plt.legend(loc='upper left')
# plt.show()6

# ################Perceptron algorithm test area###################


# ppn = PerceptronAlgorithm.Perceptron(eta = 0.1, n_iter = 6)
# ppn.fit(X,y)
# # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# # plt.xlabel('Epochs')
# # plt.ylabel('Number of misclassifications')
# # plt.show()

# plot_decision_regions(X, y, classifier=ppn)
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()

############################## for AdalineAlgorithm##################
# # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# # ada1 = AdalineAlgorithm.AdalineAlgorithm(eta = 0.01, n_iter=10)
# # ada1.fit(X_train, y_train)
# # ax[0].plot(range(1, len(ada1.cost_) + 1),np.log10(ada1.cost_), marker='o')
# # ax[0].set_xlabel('Epochs')
# # ax[0].set_ylabel('log(Sum-squared-error)')
# # ax[0].set_title('Adaline - Learning rate 0.01')
# # ada2 = AdalineAlgorithm.AdalineAlgorithm(n_iter=10, eta=0.0001).fit(X_train, y_train)
# # ax[1].plot(range(1, len(ada2.cost_) + 1),ada2.cost_, marker='o')
# # ax[1].set_xlabel('Epochs')
# # ax[1].set_ylabel('Sum-squared-error')
# # ax[1].set_title('Adaline - Learning rate 0.0001')
# # plt.show()

# X_std = np.copy(X_train)
# X_std[:,0] = (X_train[:,0] - X_train[:,0].mean())/X_train[:,0].std()
# X_std[:,1] = (X_train[:,1] - X_train[:,1].mean())/X_train[:,1].std()
# ada = AdalineAlgorithm.AdalineAlgorithm(n_iter=15, eta=0.01)
# ada.fit(X_std,y_train)
# plot_decision_regions(X_std,y_train, classifier = ada)
# plt.title('Adaline - Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')
# plt.show()
# plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Sum-squared-error')
# plt.show()


#################### for Adaline stochastic gradient descent################
X_std = np.copy(X_train)
X_std[:,0] = (X_train[:,0] - X_train[:,0].mean())/X_train[:,0].std()
X_std[:,1] = (X_train[:,1] - X_train[:,1].mean())/X_train[:,1].std()
ada = AdalineSGDAlgorithm.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y_train)
plot_decision_regions(X_std, y_train, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()