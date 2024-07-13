# CIS662 HW6
# Gina Roh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


def get_category(ratio):
    if ratio < 1.05:
        return 0
    elif ratio > 1.15:
        return 2
    else:
        return 1

def linear_regression(train_input, train_target, test_input, test_target):
    lr = LinearRegression()

    # A linear regression model minimizes the mean squared error on the training set. 
    # This means that the parameters obtained after the fit (i.e. coef_ and intercept_) 
    # are the optimal parameters that minimizes the mean squared error.
    # Source: https://inria.github.io/scikit-learn-mooc/python_scripts/linear_regression_in_sklearn.html
    lr.fit(train_input, train_target)
    # print(lr.score(train_input, train_target))
    # print(lr.score(test_input, test_target))
    result = lr.predict(test_input)
    return result

def nn_regression(train_input, train_target, test_input, test_target, learning_rate):
    # Build 1-hidden layer neural network (5-3-1 architecture). 
    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='relu', input_shape=(5, ), name='hidden'))
    model.add(keras.layers.Dense(1, activation='relu', name = 'output'))
    model.summary()

    # Make design decisions: node functions, data normalization, output interpretation, optimizer choice, etc.
    # refer to https://www.tensorflow.org/tutorials/keras/regression
    model.compile(loss='mean_absolute_error',
                optimizer=keras.optimizers.Adam(float(learning_rate)))

    model.fit(train_input, train_target, epochs=100, verbose=0)
    model.evaluate(test_input, test_target, verbose=0)

    # predict result with the model.
    result = model.predict(test_input, verbose=0)

    return result.reshape(20,)

def logistic_regression(train_input, train_target, test_input, test_target):
    lr = LogisticRegression(C=20, max_iter=1000)
    lr.fit(train_input, train_target)
    # print(lr.score(train_input, train_target))
    # print(lr.score(test_input, test_target))
    result = lr.predict(test_input)
    prob = lr.predict_proba(test_input)
    accuracy = lr.score(test_input, test_target)

    return prob, result, accuracy

def nn_classification(train_input, train_target, test_input, test_target, learning_rate):
    # Build 1-hidden layer 6-6-3 neural network.
    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='relu', input_shape=(6, ), name='hidden'))
    model.add(keras.layers.Dense(3, activation='softmax', name = 'output'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.01), 
                  loss='sparse_categorical_crossentropy', 
                  metrics='accuracy')
    
    # Train model.
    history = model.fit(train_input, train_target, epochs=200, verbose=0)

    # Evaluate the results on the remaining (20%) test data.
    _, accuracy = model.evaluate(test_input, test_target)
    prob = model.predict(test_input, verbose=0)
    result =  np.argmax(prob, axis=1)
    return prob, result, accuracy

def main():
    # Read data from csv
    df = pd.read_csv('51-60.csv')
    
    # divide data into training sets, and testing sets
    df_training = df.iloc[20:100, :]
    df_test = df.iloc[0:20, :]
    
    ##### Task 1. Linear regression #####
    # Fit a line to go very near the 2017-2021 citaion columns, minimizing MSE.
    # Use that line to predict the 2022 cita=on numbers, as in HW4.
    train_input_1 = df_training.iloc[:, 3:8].to_numpy()
    train_target_1 = df_training.iloc[:, 8:9].to_numpy().reshape(80,)

    test_input_1 = df_test.iloc[:, 3:8].to_numpy()
    test_target_1 = df_test.iloc[:, 8:9].to_numpy().reshape(20,)
    
    # Apply linear regression and NN.
    result_lr = linear_regression(train_input_1, train_target_1, test_input_1, test_target_1)
    result_nn = nn_regression(train_input_1, train_target_1, test_input_1, test_target_1, 0.01)
    df_test['predict_LR'] = result_lr
    df_test['predict_NN'] = result_nn
    mse_lr = mean_squared_error(test_target_1, result_lr)
    mse_nn = mean_squared_error(test_target_1, result_nn)

    # Print results
    print("---------------------------------------------------------------------")
    print("Task1. Regression")
    print(df_test)
    print('MSE_LR: %.2f' % mse_lr)
    print('MSE_NN: %.2f' % mse_nn)
    print("---------------------------------------------------------------------")

    # Save result to csv
    df_test.to_csv('result1.csv', index=False)

    ##### Task2. Logistic regression #####
    # Classify individuals into 3 categories, as in HW5.
    df_2 = df.copy()

    # Calculate the ratio of (citations in 2022)/(citations in 2021).
    df_2 = df_2.assign(ratio=lambda x: round(x['cit_2022']/x['cit_2021'],2))
    df_2['category'] = df_2['ratio'].apply(get_category)
    
    # Divide data into training sets, and testing sets
    df_training_2 = df_2.iloc[20:100, :]
    df_test_2 = df_2.iloc[0:20, :]
    
    train_input_2 = df_training_2.iloc[:, 3:9].to_numpy()
    train_target_2 = df_training_2['category'].to_numpy().reshape(80,)

    test_input_2 = df_test_2.iloc[:, 3:9].to_numpy()
    test_target_2 = df_test_2['category'].to_numpy().reshape(20,)

    # Apply logistic regression and NN classification.
    prob_lr, r_lr, accuracy_lr = logistic_regression(train_input_2, train_target_2, test_input_2, test_target_2)
    prob_nn, r_nn, accuracy_nn = nn_classification(train_input_2, train_target_2, test_input_2, test_target_2, 0.01)

    df_test_2['predict_LR'] = r_lr
    df_test_2['predict_NN'] = r_nn

    # Print results
    print("---------------------------------------------------------------------")
    print("Task2. Classification")
    print(df_test_2)
    print('Accuracy_LR: %.2f' % accuracy_lr)
    print('Accuracy_NN: %.2f' % accuracy_nn)
    print("---------------------------------------------------------------------")

    # Save result to csv
    df_test_2.to_csv('result2.csv', index=False)

if __name__=="__main__":
    main()
