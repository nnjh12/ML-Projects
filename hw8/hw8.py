# CIS662 HW7
# Gina Roh

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics

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

def get_percent_change(x1, x2):
    return (x2 - x1) / x1

def main():
    # Read data from csv
    df = pd.read_csv('51-60.csv')
    
    # Calculate the ratio of (citations in 2022)/(citations in 2021).
    df = df.assign(ratio=lambda x: round(x['cit_2022']/x['cit_2021'],2))
    df['category'] = df['ratio'].apply(get_category)
    
    # divide data into training sets, and testing sets
    df_training = df.iloc[20:100, :]
    df_test = df.iloc[0:20, :]
    
    train_input = df_training.iloc[:, 3:9].to_numpy()
    train_target = df_training['category'].to_numpy().reshape(80,)

    test_input = df_test.iloc[:, 3:9].to_numpy()
    test_target = df_test['category'].to_numpy().reshape(20,)
    
    # Calssification Using Adaboost.
    # Use the same dataset that you used for HW5 for the classification
    # Apply Adaboost to the data set.

    # Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=500, learning_rate=1)

    # Train Adaboost Classifer
    abc.fit(train_input, train_target)

    #Predict the response for test dataset
    predict = abc.predict(test_input)
    df_test['predict'] = predict

    print('-----part3-----')
    print(df_test.iloc[:, np.r_[0:9, 12:14]])

    print('Training score: ',abc.score(train_input, train_target))
    print('Test score: ',abc.score(test_input, test_target))

    # Introduce 5 new features based on the cita7on numbers. and use them in the RF instead of the citation numbers directly.
    # Each new feature is:
    # ((citation number in year n+1)-(citation number in year n))/(cita7on number in year n)
    # for 2016<n<2022.
    df2 = df.copy()
    df2 = df2.assign(change18=lambda x: (x['cit_2018']/x['cit_2017'] - 1))
    df2 = df2.assign(change19=lambda x: (x['cit_2019']/x['cit_2018'] - 1))
    df2 = df2.assign(change20=lambda x: (x['cit_2020']/x['cit_2019'] - 1))
    df2 = df2.assign(change21=lambda x: (x['cit_2021']/x['cit_2020'] - 1))
    df2 = df2.assign(change22=lambda x: (x['cit_2022']/x['cit_2021'] - 1))

    # divide data into training sets, and testing sets
    df_training2 = df2.iloc[20:100, :]
    df_test2 = df2.iloc[0:20, :]
    
    train_input2 = df_training2.iloc[:, 13:18].to_numpy()
    train_target2 = df_training2['category'].to_numpy().reshape(80,)

    test_input2 = df_test2.iloc[:, 13:18].to_numpy()
    test_target2 = df_test2['category'].to_numpy().reshape(20,)

    # Create adaboost classifer object
    abc2 = AdaBoostClassifier(n_estimators=500, learning_rate=1)

    # Train Adaboost Classifer
    abc2.fit(train_input2, train_target2)

    #Predict the response for test dataset
    predict = abc2.predict(test_input2)
    df_test2['predict'] = predict

    print('-----part4-----')
    print(df_test2.iloc[:, np.r_[0:3, 13:18, 12, 18]])


    print('Training score: ',abc2.score(train_input2, train_target2))
    print('Test score: ',abc2.score(test_input2, test_target2))


if __name__=="__main__":
    main()