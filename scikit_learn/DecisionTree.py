from sklearn.svm import SVC
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris() # load 150 samples
X = iris.data[:,[2,3]]      # take the feature in collunm 2 and 3 of data
y = iris.target             

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state =0)
sc = StandardScaler()
sc.fit(X_train)                     # the StandardScaler eastimated the parameters mean and standard deviation for the feature dimension from the traning data
X_train_std = sc.transform(X_train) # statndardized the training data using the estimated parameters mean and standard deviation above
X_test_std = sc.transform(X_test)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

def plot_decision_regions(X,y, classifier, test_idx = None, resolution = 0.02):
    
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',alpha=1.0, linewidths=1, marker='o',s=55, label='test set')

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state = 0)
tree.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined,classifier=tree, test_idx  =range(105,150))
plt.show()