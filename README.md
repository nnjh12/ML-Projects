# ML-Projects
ML Projects based on CIS662 homework

## HW2 (Principal Components Analysis)
> We address 3 different prediction problems using ML:
> * Predicting 2022 citation numbers using the university rank and 2017-2021 citation numbers.
> * Predicting the h-index using the university rank and all (2017-2022) citation numbers.
> * Predicting the i10-index using the university rank and all (2017-2022) citation numbers.
> Use Principal Components Analysis on the data set.

## HW3 (Nearest neighbor)
Predict 2022 citations based on all the 2017-2021 citations. Use an appropriate distance measure to determine the nearest neighbors.  
What is the right number of clusters for this problem? Why?  
Tabulate the following predictions for the 2022 citation numbers for the test set:
* same as the 2022 citation number of the nearest neighbor from the training set;
* same as the point nearest the cluster centroid;
* average of all others from the training set in the same cluster.

## HW4 (Neural network)
Predict 2022 citations based on all the 2017-2021 citations. Try a 1-hidden layer neural network (5-3-1 architecture) using the backpropagation algorithm.  
Play with different values of the learning rate to see what works best for this problem.

## HW5 (Classification using NN)
Classification problem using a 1-hidden layer 6-6-3 neural network.  
Based on the ratio (citations in 2022)/(citations in 2021), approximated to two decimal places, determine the category of each individual as one of the three shown below:
* Low (<1.05).
* Medium (1.06-1.15).
* High (>1.15).
The inputs to the network would be the citation numbers from 2017 to 2022, normalized as you consider appropriate.

## HW6 (Linear regression and Logistic regression)
Linear regression: Fit a line to go very near the 2017-2021 citation columns, minimizing MSE. Use that line to predict the 2022 citation numbers.  
Logistic regression: Classify individuals into 3 categories, as in HW5.

## HW7 (Random forest)
### Part-1:
Random forest approach for the same classification problem as HW5.  
(It probably won't perform well; say why.)
### Part-2:
Introduce 5 new features based on the citation numbers. Use them in the RF instead of the citation numbers directly.  
Each new feature is:
```
((cita7on number in year n+1)-(cita7on number in year n))/(cita7on number in year n) for 2016<n<2022.
```

## HW8 (Adaboost)
Adaboost for the same classification problem as HW5.

