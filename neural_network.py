import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
import pandas as pd  # data processing


class Model2(Model):
    def __init__(self):
        super(Model2, self).__init__()
        self.flatten = Flatten()
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        self.d2 = Dense(2, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y


class Model1(Model):
    def __init__(self):
        super(Model1, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(2, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y


def neural_network(X,y,X_test,test_data):
    #model=Model2()
    model=Model1()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    model.fit(X, y, batch_size=32, epochs=3, validation_freq=10)
    predictions = model.predict(X_test)
    predictions = tf.argmax(predictions, 1)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('NN.csv', index=False)