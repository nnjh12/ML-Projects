# CIS662 HW2
# Gina Roh
# Refer to https://datascienceplus.com/principal-component-analysis-pca-with-python/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def find_quantile_threshold(df, col_name):
    p1 = df[col_name].quantile(0.25)
    p2 = df[col_name].quantile(0.5)
    p3 = df[col_name].quantile(0.75)
    return p1, p2, p3

def find_quantile(p1, p2, p3):
    return lambda v : 'red' if v <= p1 else ( 'orange' if v <= p2 else ( 'green' if v <= p3 else 'blue'))

def divide_quantile(df, col_name):
    p1, p2, p3 = find_quantile_threshold(df, col_name)
    column = df[col_name].apply(find_quantile(p1, p2, p3))
    return column

def scatter_quantile_pca(df, input_cols, output_col): 
    # Pre-processing.
    df_copy = df.loc[:, df.columns[input_cols]]
    scaler = StandardScaler()
    scaler.fit(df_copy)
    scaled_data = scaler.transform(df_copy)

    # Create two PCA objects.
    pca = PCA(n_components=2)
    pca.fit(scaled_data)

    # Transform data to 2 principal components.
    x_pca = pca.transform(scaled_data)
    print(scaled_data.shape)
    print(x_pca.shape)
    # print(x_pca)

    # Divide the data into 4 quarters based on the variable to be predicted, and append to result.
    result = pd.DataFrame(x_pca, columns=['x1', 'x2'])
    result[output_col] = divide_quantile(df, output_col)
    # print(result)

    # Generate a scatter plot for all data using these 4 colors, whose axes are the two most important principal components.
    plt.figure(figsize=(8,6))
    s = plt.scatter(result['x1'], result['x2'], c=result[output_col])
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    plt.title('Scatter plot for ' + output_col)
    plt.grid(True)
    plt.show()

def main():
    df = pd.read_csv('51-60.csv')
    # df = pd.read_csv('51-60_copy.csv') #The dataset after removing row2(outlier)
    print(df)

    # Problem1. Predicting 2022 citation numbers using the university rank and 2017-2021 citation numbers.

    scatter_quantile_pca(df, [0,3,4,5,6,7], 'cit_2022')
    # Problem2. Predicting the h-index using the university rank and all (2017-2022) citation numbers.
   
    scatter_quantile_pca(df, [0,3,4,5,6,7,8], 'h_index')
    
    # Problem3. Predicting the i10-index using the university rank and all (2017-2022) citation numbers.
    scatter_quantile_pca(df, [0,3,4,5,6,7,8], 'i_10_index')

if __name__=="__main__": 
    main() 