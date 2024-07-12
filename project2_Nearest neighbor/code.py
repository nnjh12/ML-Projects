# CIS662 HW3
# Gina Roh
# https://www.statology.org/k-means-clustering-in-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def find_cluster(x, centroids): #x: one data point, centroids: list of centroids
    dist = []
    for i in range(len(centroids)):
        dist_ind = np.sqrt(np.sum((x.to_numpy() - centroids[i])**2, axis=0))
        dist.append(dist_ind)
    return np.argmin(dist) 

def find_nearest_neighbors(x_df, y_df): #x_df: test datasets, y_df: training datasets
    nearest_neighbors = []
    x_numpy = x_df.to_numpy()
    y_numpy = y_df.to_numpy()
    
    for i in range (len(x_numpy)):
        dist = []
        for j in range(len(y_numpy)):

            #Computing Euclidean distance
            dist_ind = np.sqrt(np.sum((x_numpy[i] - y_numpy[j])**2, axis=0)) 
            dist.append(dist_ind)
        nearest_neighbors.append(np.argmin(dist))
    return nearest_neighbors

def main():
    # read data from csv
    df = pd.read_csv('51-60.csv')
    
    # divide data into training sets, and testing sets
    df_training_original = df.iloc[20:100, :]
    df_test_original = df.iloc[0:20, :]

    df_training = df_training_original.iloc[:, 3:8]
    df_test = df_test_original.iloc[:, 3:8]

    # initialize kmeans parameters
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "random_state": 1,
    }

    # create list to hold SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df_training)
        sse.append(kmeans.inertia_)

    # visualize results
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    # instantiate the k-means class, using optimal number of clusters
    kmeans = KMeans(init="random", n_clusters=3, n_init=10, random_state=1)

    # fit k-means algorithm to data
    kmeans.fit(df_training)

    # view cluster assignments for each observation
    centroids = kmeans.cluster_centers_

    # append cluster assingments to original DataFrame
    df_training_original['cluster'] = kmeans.labels_

    # find cluster for every data points in test data
    clusters = []
    for i in range(len(df_test)):
        clusters.append(find_cluster(df_test.iloc[i], centroids))
    df_test_original['cluster'] = clusters

    # Prediction1. same as the 2022 citation number of the nearest neighbor from the training set
    nearest_neighbors = find_nearest_neighbors(df_test, df_training)
    cit_2022_nn = [df_training_original['cit_2022'].iloc[x] for x in nearest_neighbors]
    df_test_original['cit_2022_p1'] = cit_2022_nn
    
    # Prediction2. same as the point nearest the cluster centroid
    cit_2022_centroid = []
    
    for i in range(len(centroids)):
        dist = []
        temp = df_training_original[df_training_original['cluster'] == i]
        temp_numpy = temp.iloc[:, 3:8].to_numpy()
        centroid = centroids[i]
        for j in range(len(temp)):
            dist_ind = np.sqrt(np.sum((temp_numpy[j] - centroid)**2, axis=0)) 
            dist.append(dist_ind)
        cit_2022_centroid.append(temp['cit_2022'].iloc[np.argmin(dist)])
    df_test_original['cit_2022_p2'] = df_test_original['cluster'].apply(lambda x : cit_2022_centroid[x])
    
    
    # Prediction3. average of all others from the training set in the same cluster.
    cit_2022_average = []
    for i in range(3):
        temp = df_training_original[df_training_original['cluster'] == i]
        cit_2022_average.append(temp["cit_2022"].mean())
    df_test_original['cit_2022_p3'] = df_test_original['cluster'].apply(lambda x : cit_2022_average[x])
    
    # Calculate the error of each prediction.
    errors = []
    for i in range(3):
        err = np.average(np.absolute(df_test_original['cit_2022'].to_numpy() - df_test_original.iloc[:, i + 12 ].to_numpy()))
        errors.append(err)

    # Show result
    print("----- CLUSTERS INFO -----")
    for i in range(3):
        print("cluster%d" %(i + 1))
        print("    centroid: %s" % centroids[i])
        print("    value of cit_2022 of nearest data point from centroid: %d" %cit_2022_centroid[i])
        print("    average of cit_2022 in cluster%d: %.2f" %(i + 1, cit_2022_average[i]))

    print("----- RESULT -----")  
    print(df_test_original)
    for i in range(3):
        print("Average difference of prediction %d: %.2f" % (i + 1, errors[i]))

if __name__=="__main__":
    main()