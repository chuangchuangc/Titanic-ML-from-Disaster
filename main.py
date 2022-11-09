import pandas as pd  # data processing
import selection
import classify_method
import neural_network


def preprocessing(data):
    # Replace NaN with mean of values in the same column
    data['Age'].fillna(value=data['Age'].mean(), inplace=True)
    # male represent 1, female for 0
    data['Sex'].replace(['male', 'female'], [1, 0], inplace=True)
    data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    data['Fare'].fillna(value=data['Fare'].mean(), inplace=True)
    # Replace NaN with mean of values in the same column
    data['Embarked'].fillna(value=data['Embarked'].mean(), inplace=True)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
preprocessing(train)
preprocessing(test)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])
y = train["Survived"]

features1 = selection.forward_regression(X, y, 0.05)
X = pd.get_dummies(train[features1])
X_test = pd.get_dummies(test[features1])

classify_method.randomForest(X, y, X_test, test)
classify_method.svc(X, y, X_test, test)
classify_method.logistic(X, y, X_test, test)
classify_method.knn(X,y,X_test, test)
#neural_network.neural_network(X, y, X_test, test)


