# ML-Projects
Worked on prediction and classification problems using datasets that included university rankings, citation numbers from 2017-2022, h-index, and i10-index.
These projects were part of CIS662 coursework at Syracuse University.

## Project1: Principal Components Analysis
Addressed three different prediction problems using ML:

* Predicted 2022 citation numbers using university rank and 2017-2021 citation numbers.
* Predicted the h-index using university rank and citation numbers from 2017-2022.
* Predicted the i10-index using university rank and citation numbers from 2017-2022.

Used Principal Component Analysis (PCA) on the dataset.

<b>[Results](./project1_PCA/report.pdf)</b><br />
The first principal component(PC1) effectively predicted the 2022 citation numbers/h-index/i10-index, while the second principal component(PC2) did not contribute significantly to the prediction. The values associated with the PC2 were scattered and showed mixed results in relation to the predictions.

## Project2: Nearest neighbor
Using an appropriate distance measure to determine the nearest neighbors, predicted 2022 citations based on the 2017-2021 citations.
Used the following predictions for the 2022 citation numbers for the test set:

1) The same as the 2022 citation number of the nearest neighbor from the training set.
1) The same as the point nearest the cluster centroid.
1) The average of all others from the training set in the same cluster.

<b>[Results](./project2_NearestNeighbor/report.pdf)</b><br />
> Prediction1: loss: 866.80<br />
> Prediction2: loss: 961.10<br />
> Prediction3: loss: 940.43

The prediction method that uses the value closest to the 2022 citation number from the training set's nearest neighbor has the smallest difference from the actual 2022 citation value in the test datasets.

## Project3: Neural network
Predicted 2022 citations based on all the 2017-2021 citations. Used a 1-hidden layer neural network (5-3-1 architecture) with the backpropagation algorithm. 
Various values of the learning rate were experimented with to determine the most effective approach for this problem.

#### Model
* 1 hidden layer 5-3-1 neural network.
* Activation function of Hidden layer: ReLU
* Loss function: mean_absolute_error
* Optimizer: adam with learning rate 0.1
* Evaluate metrics: Loss

<b>[Results](./project3_NeuralNetwork/report.pdf)</b><br />
> Test results: loss: 99.12

I found that the neural network (NN) significantly outperformed the other three predictions from Project2. In this experiment, using a learning rate of 0.1, the NN achieved a loss (mean of absolute errors) of 99.12. In comparison, the previous losses were much higher at 866.80, 961.10, and 940.43.

## Project4: Classification using NN
Utilized a 1-hidden layer 6-6-3 neural network for the classification task.
Based on the ratio (citations in 2022)/(citations in 2021), the category of each individual was determined as one of the following:

* Low (<1.05).
* Medium (1.06-1.15).
* High (>1.15).

The inputs to the network were the citation numbers from 2017 to 2022, appropriately normalized.

#### Model
* 1 hidden layer 6-6-3 neural network.
* Activation function of Hidden layer: ReLU
* Activation function of Output layer: Softmax
* Loss function: sparse_categorical_crossentropy
* Optimizer: adam with learning rate 0.01
* Evaluate metrics: Loss and Accuracy

<b>[Results](./project4_ClassificationNN/report.pdf)</b><br />
> Training results: loss: 0.3850, accuracy: 0.8625 (Upon reaching the 200th epoch)<br />
> Test results: loss: 0.7344, accuracy: 0.7500

## Project5: Linear regression and Logistic regression
### Part-1:
Linear regression: Fitted a line to closely approximate the 2017-2021 citation columns, minimizing Mean Squared Error (MSE). This line was then used to predict the 2022 citation numbers.

<b>[Results](./project5_LinearRegressionLogisticRegression/report.pdf)</b><br />
> Mean Square Error: 22,318

The linear regression (LR) model has a mean square error of 22,318. It's surprisingly effective despite its simplicity, delivering consistently strong performance.

### Part-2:
Logistic regression: Individuals were classified into 3 categories, as in project.

<b>[Results](./project5_LinearRegressionLogisticRegression/report.pdf)</b><br />
> Accuracy: 1.0

The logistic regression model achieves a remarkable 100% accuracy when classifying the test set, which is impressive given its simplicity.

## Project6: Random forest
Random forest was employed for the same classification problem as in project4.
Five new features were used in the random forest model instead of the citation numbers directly.
Each new feature is:
`((citation number in year n+1)-(citation number in year n))/(citation number in year n)` for 2016<n<2022.

<b>[Results](./project6_RandomForest/report.pdf)</b><br />
> Training Score: 1.0<br />
> Test Score: 1.0

The random forest model using new features achieved perfect accuracy.

## Project7: Adaboost
Adaboost was employed for the same classification problem as in project6.

<b>[Results](./project7_Adaboost/report.pdf)</b><br />
> Training Score: 1.0<br />
> Test Score: 1.0

The Adaboost model using new features achieved perfect accuracy.


