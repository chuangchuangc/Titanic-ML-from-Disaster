from sklearn.ensemble import RandomForestClassifier
import pandas as pd  # data processing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def knn(X, y, X_test, test):
    # max number for n
    number_max = 40
    # store number
    number_k = []
    # store performance
    perfromance_k = []
    max_index = 1
    max = 0
    # get different result of different neighbors number
    for i in range(1, number_max):
        knn = KNeighborsClassifier(n_neighbors=i)
        # use accurancy as performance
        scores = cross_val_score(knn, X, y, cv=6, scoring='accuracy')
        # use mean as score
        score = scores.mean()
        # append into list
        perfromance_k.append(score)
        # get max
        if (max < score):
            max_index = i
            max = score
        number_k.append(i)
    # plot
    print("max index:" + str(max_index))
    plt.plot(number_k, perfromance_k)
    plt.title("performance of the classifier versus the number of neighbors")
    plt.xlabel('number of neighbors')
    plt.ylabel('Cross-Validation Accuracy')
    plt.show()

    model = KNeighborsClassifier(n_neighbors=max_index)
    model.fit(X, y)
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
    output.to_csv('knn_result.csv', index=False)


def randomForest(X, y, X_test, test):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
    output.to_csv('randomForest_result.csv', index=False)


def logistic(X, y, X_test, test):
    model = LogisticRegression()
    model.fit(X, y)
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
    output.to_csv('LogisticRegression_result.csv', index=False)


def svc(X, y, X_test, test):
    model = svm.SVC()
    model.fit(X, y)
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
    output.to_csv('svc_result.csv', index=False)


