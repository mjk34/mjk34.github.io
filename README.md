# MAP6990 Capstone Project - Fraud Detection

### Intro

With impact of 2020's global pandemic and the transition to online learning, work and shopping has increased. With the increase of online activity, we can also see a rise in threat actors and malicious activities, among them- fraudulent credit card transactions from stolen credentials. The goal of this project is to explore credit card fraud detection using machine learning classification models.

The dataset I will work with comes from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

This dataset consists of highly unbalanced credit card transactions from European cardholders from 2013. The features are that of PCA transformation, time, amount, and class (whether its legitimate or fraudulent). The author suggests to measure accuracy via Area Under the Precision-Recall Curve (AUPRC) and that the confusion matrix accuracy is not sufficient for unbalanced classification


### Importing the Data

First we imported the data with pandas read_csv from our locally downloaded csv from the link above: 
[display head() results]

[display discribe() results]


### Check for Missing Values

Checked for null or missing values both manually (nested loop) and using pandas dataframe isnull sum. In both checks, I was not able to find any missing values, they should be good for processing.

[display results]


### Partition the Data

In this project, I will split the dataset into 70% training and 30% testing and normalize the training set using scikit's standardscaler. Related works on this dataset have sometimes used 80-20 split, this could result in better performance readings, but I like the 70-30 split in general as it can avoid unbalanced splits. In this case the data is already very skewed.


### Dimensionality Reduction

Due to the PCA features, there are an overwelming number of features/dimensions, we would normally want to reduce its size and select the most important features to improve runtime and performance. The problem with using eigenvalues or PCA is that Features V1, V2, … V28 are the result of PCA transformation and the amount of information/meaning between those features is equally spread out. This can be seen below:

[display variance graph]

One solution I've seen in order to select features is the use of the feature importance function from the Random Forest Classifier for making accurate predictions. Feature importance looks into how much each feature contributes to the overall accuracy of the model and calculates a score for each feature based on the number of times its used in the decision making process. Here's a look at the feature importance scores for our dataset:

[display feature importance graph]

Based on RandomForest Feature Importance, should we take the first 6 features: V17, V12, V14, V10, V16, V11. We can retain above 70% of the information using only 1/4th of the features.


### Modeling

For machine learning models, we will look to optimize **Logistic Regression** for model comparisons. *Logistic regression is great for binary classification, in this case: whether the transaction was fraudulent or legitimate.*


Additional models that might be of interest for exploration may be:

**Random Forest**: *We utilized part of the random forest classifier for feature selection, overall this model is really powerful*

**SVM**: *A common classification model and works on feature values that are complicated and difficult to linearly separate*

**Gradient Boosting**: *A popular model that can handle a large dataset and provides high accuracy.

**Neural Networks**: *A complex model that can provide high accuracy*


All Feature Train

    Logistic Regression Train Time: 0.646 seconds
    Suppor Vector Machine Train Time: 75.380 seconds
    Random Forest Train Time: 4.765 seconds

Selected Feature Train

    Logistic Regression Train Time: 0.247 seconds
    Suppor Vector Machine Train Time: 190.861 seconds
    Random Forest Train Time: 1.907 seconds

From the results above, we can show the runtime performance between the all the features and the selected features:

    Logistic Regression: 61.76% decrease in runtime
    Support Vector Machine: 153.19% increase in runtime
    Random Forest: 59.98% decrease in runtime

Both Logistic Regression and Random Forest improved in runtime with the selected feature split, inversely SVM increased in runtime due to fewer features because the training algorithm computes the distance betweeen each data point and decision boundary- with the fewer features, the distance calculation requires more time and resource, slowing down the training process

### Testing

Given the class imbalance ratio, I will use the author's recommend measurment for accuracy using the Area Under the Precision-Recall Curve (AUPRC).

All Feature Test

    Logistic Regression:    Area: 0.753    Miss: 68    Accuracy: 0.9992
                    SVM:    Area: 0.857    Miss: 42    Accuracy: 0.9995
          Random Forest:    Area: 0.824    Miss: 50    Accuracy: 0.9994
          
Selected Feature Test

    Logistic Regression:    Area: 0.700    Miss: 81    Accuracy: 0.9990
                    SVM:    Area: 0.835    Miss: 47    Accuracy: 0.9994
          Random Forest:    Area: 0.824    Miss: 50    Accuracy: 0.9994
          
From the model testing results above, we can see that:

Logistic Regression had a 7.015% decrease in AUC and 13 more missclassifications for the trade off of shaving 0.10 seconds (~55% of the runtime) between all of the features and the selected features

SVM had a 2.557% decrease in AUC and 5 more missclassifications for the trade off of shaving 15.331 seconds (~82% of the run time) between all of the features and the selected features

Random Forest had the same performance for AUC and accuracy with the improvement of 0.08 seconds (~9% of the run time) between all of the features and the selected features

For the models we learned in the course (Logistic Regression, SVM, and Random Forest), we can see that both SVM and Random Forest were comparible in terms of AUPRC and Accuracy Results. SVM had the best performance for the highest amount of resources (features and runtime) while Random Forest was a close in AUPRC and Accuracy with much better runtime even with only the selected features.

### Additional Models

n additional to the models we learned in the course, I would like to test the capabilities of gradient boosting and neural network, these features will require more resources than our other models and push for the best accuracy. When working with financial transactions, it would be much better to maximize the true positive and minimize the false positives to avoid disabling someones payment method on a missclassification.

All Feature Train

    Gradient Boosting:    Area: 0.789    Miss: 60    Accuracy: 0.9993    Train Time: 192.863 seconds
                NNMLP:    Area: 0.836    Miss: 47    Accuracy: 0.9995    Train Time: 0.030 seconds

Selected Feature Train

    Gradient Boosting:    Area: 0.748    Miss: 75    Accuracy: 0.9991    Train Time: 39.179 seconds
                NNMLP:    Area: 0.789    Miss: 59    Accuracy: 0.9993    Train Time: 0.018 seconds
                
The Multi-layered Perceptron Neural Network Model was incredibly surprising, the model was super fast in training and testing while producing really good AUPRC Area and Low Missclassification in the all feature test. A downside of this model is that the results of the tests are a bit volatile, reruns of the same code could change the performance by 3-7 missclassifications.

### Results

For model testing, the author of the credit card fraud detection data recommended using AUPRC because the confusion matrix accuracy is not meaningful for unbalanced classification. In addition we wanted to highlight the accuracy and missclassification of the models as false possitives could have large consequences in the real world.

Overall, each model hyperparameter was selected to optimize the area and accuracy. The hyperparameters were kept the same across the same model to measure accuracy vs runtime between the full feature and selected feature splits.

Based on our testings above, SVM and Neural Network models had the highest AUPRC and lowest missclassifications on the full features, and the Neural Network model blitzed SVM runtime in both training and testing.

In the selected feature testing, Random Forest performed the best and was able to capitalize on the runtime improvements while maintaining the good AUPRC and accuracy. Part of the Random Forest performing consistently especially in the selected feature test could be correlated with using the Random Forest Classifier during the important feature selection earlier.

