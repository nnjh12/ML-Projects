# CIS662 HW5
# Gina Roh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Based on the ratio (citations in 2022)/(citations in 2021), approximated to two decimal places, 
# determine the category of each individual as one of the three shown below:
# 1. Low (<1.05).
# 2. Medium (1.06-1.15).
# 3. High (>1.15).
def get_category(ratio):
    if ratio < 1.05:
        return 0
    elif ratio > 1.15:
        return 2
    else:
        return 1

def main():
    # Read data from csv
    df = pd.read_csv('51-60.csv')

    # Calculate the ratio of (citations in 2022)/(citations in 2021).
    df = df.assign(ratio=lambda x: round(x['cit_2022']/x['cit_2021'],2))
    df['category'] = df['ratio'].apply(get_category)
    
    # Divide data into training sets, and testing sets
    df_training = df.iloc[20:100, :]
    df_test = df.iloc[0:20, :]
    
    train = df_training.iloc[:, 3:9].to_numpy()
    train_target = df_training['category'].to_numpy().reshape(80,)

    test = df_test.iloc[:, 3:9].to_numpy()
    test_target = df_test['category'].to_numpy().reshape(20,)

    print(train.shape, train_target.shape, test.shape, test_target.shape)

    # Build 1-hidden layer 6-6-3 neural network.
    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='relu', input_shape=(6, ), name='hidden'))
    model.add(keras.layers.Dense(3, activation='softmax', name = 'output'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.01), 
                  loss='sparse_categorical_crossentropy', 
                  metrics='accuracy')
    
    # Train model.
    history = model.fit(train, train_target, epochs=200)
    plt.plot(history.history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    # Evaluate the results on the remaining (20%) test data.
    model.evaluate(test, test_target)
    result = model.predict(test, verbose=0)

    df_test['predict'] = np.argmax(result, axis=1)
    print(df_test.iloc[:, np.r_[0:9, 12:14]])


if __name__=="__main__":
    main()