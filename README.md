# ML-Projects
ML Projects based on CIS662 coursework at Syracuse University.

## Project1: Principal Components Analysis
Addressed three different prediction problems using ML:

* Predicted 2022 citation numbers using university rank and 2017-2021 citation numbers.
* Predicted the h-index using university rank and citation numbers from 2017-2022.
* Predicted the i10-index using university rank and citation numbers from 2017-2022.

Used Principal Component Analysis (PCA) on the dataset.

## Project2: Nearest neighbor
Using an appropriate distance measure to determine the nearest neighbors, predicted 2022 citations based on the 2017-2021 citations.
Used the following predictions for the 2022 citation numbers for the test set:

* The same as the 2022 citation number of the nearest neighbor from the training set.
* The same as the point nearest the cluster centroid.
* The average of all others from the training set in the same cluster.

## Project3: Neural network
Predicted 2022 citations based on all the 2017-2021 citations. Used a 1-hidden layer neural network (5-3-1 architecture) with the backpropagation algorithm. 
Various values of the learning rate were experimented with to determine the most effective approach for this problem.

## Project4: Classification using NN
Utilized a 1-hidden layer 6-6-3 neural network for the classification task.
Based on the ratio (citations in 2022)/(citations in 2021), the category of each individual was determined as one of the following:

* Low (<1.05).
* Medium (1.06-1.15).
* High (>1.15).

The inputs to the network were the citation numbers from 2017 to 2022, appropriately normalized.

## Project5: Linear regression and Logistic regression
### Part-1:
Linear regression: Fitted a line to closely approximate the 2017-2021 citation columns, minimizing Mean Squared Error (MSE). This line was then used to predict the 2022 citation numbers.
### Part-2:
Logistic regression: Individuals were classified into 3 categories, as in project4.

## Project6: Random forest
### Part-1:
Random forest was employed for the same classification problem as in project4.
### Part-2:
Five new features were used in the random forest model instead of the citation numbers directly.
Each new feature is:
> ((citation number in year n+1)-(citation number in year n))/(citation number in year n) for 2016<n<2022.

## Project7: Adaboost
Adaboost was employed for the same classification problem as in project4.

