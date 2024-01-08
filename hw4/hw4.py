# CIS662 HW4
# Gina Roh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def main():
    # read data from csv
    df = pd.read_csv('51-60.csv')
    
    # divide data into training sets, and testing sets
    df_training = df.iloc[20:100, :]
    df_test = df.iloc[0:20, :]
    
    train = df_training.iloc[:, 3:8].to_numpy()
    train_target = df_training.iloc[:, 8:9].to_numpy().reshape(80,)

    test = df_test.iloc[:, 3:8].to_numpy()
    test_target = df_test.iloc[:, 8:9].to_numpy().reshape(20,)

    print(train.shape, train_target.shape, test.shape, test_target.shape)

    # Build 1-hidden layer neural network (5-3-1 architecture). 
    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='relu', input_shape=(5, ), name='hidden'))
    model.add(keras.layers.Dense(1, activation='relu', name = 'output'))
    model.summary()

    # Play with different values of the learning rate to see what works best for this problem. 
    learning_rates = ['0.0', '0.001', '0.01', '0.1', '0.25', '0.5', '0.75', '1.0']
    losses = []
    results = []
    for lr in learning_rates:
        print(f'learning rate: {lr}')
        # reset model states
        model.reset_states()

        # Make design decisions: node functions, data normalization, output interpretation, optimizer choice, etc.
        # refer to https://www.tensorflow.org/tutorials/keras/regression
        model.compile(loss='mean_absolute_error',
                    optimizer=keras.optimizers.Adam(float(lr)))

        history = model.fit(train, train_target, epochs=100, verbose=0)
        plt.plot(history.history['loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f'learning rate: {lr}')
        plt.show()

        loss = model.evaluate(test, test_target, verbose=0)
        losses.append(loss)
        print(f'loss: {loss}')

        # predict result with the model.
        result = model.predict(test, verbose=0)
        results.append(result)

    # Plot losses for each learning rate.
    print(losses)
    fig, ax = plt.subplots()
    ax.bar(learning_rates, losses)
    ax.set_ylabel('loss')
    ax.set_title('loss of each learning rate')
    plt.show()

    # Find the best precits and append to test data sets.  
    df_test['predicts'] = results[np.argmin(np.array(losses))]
    print(df_test)

if __name__=="__main__":
    main()