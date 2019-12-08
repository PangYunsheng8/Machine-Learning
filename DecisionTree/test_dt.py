import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_regression
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split

from CART import DecisionTree


def test_DecisionTree(classifier=True, criterion="entropy"):
    np.random.seed(12345)
    n_ex = np.random.randint(2, 100)
    n_feats = np.random.randint(2, 100)
    max_depth = np.random.randint(1, 5)

    if classifier:
        # create classification problem
        n_classes = np.random.randint(2, 10)
        X, Y = make_blobs(n_samples=n_ex, centers=n_classes, n_features=n_feats, random_state=1)
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

        # initialize model
        def loss(yp, y):
            return 1 - accuracy_score(yp, y)

        mine = DecisionTree(classifier=classifier, max_depth=max_depth, criterion=criterion)
        gold = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, splitter="best", random_state=1)
    else:
        # create regeression problem
        X, Y = make_regression(n_samples=n_ex, n_features=n_feats, random_state=1)
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

        # initialize model
        criterion = "mse"
        loss = mean_squared_error
        mine = DecisionTree(criterion=criterion, max_depth=max_depth, classifier=classifier)
        gold = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, splitter="best")

    print("\tClassifier={}, criterion={}".format(classifier, criterion))
    print("\tmax_depth={}, n_feats={}, n_ex={}".format(max_depth, n_feats, n_ex))
    if classifier:
        print("\tn_classes: {}".format(n_classes))

    # fit
    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)
    mine.fit(X, Y)
    gold.fit(X, Y)

    print(X_test.shape)
    print(Y_test)
    predict = mine.predict(X_test)
    print(predict)
    print(mine.root.left.value)
    print(mine.root.right.value)

    predict_gold = gold.predict(X_test)
    print(predict_gold)


test_DecisionTree(classifier=True, criterion="gini")
