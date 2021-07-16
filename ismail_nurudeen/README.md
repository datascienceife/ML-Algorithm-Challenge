## KNN Algorithm from scratch.

**K Nearest Neighbor** is a simple supervised machine learning algorithm.

K-nearest neighbors (KNN) algorithm uses ‘feature similarity’ to predict the values of new datapoints. 
This means that the new data point will be assigned a value based on how closely it matches the points in the training set. 


In my implementation of the algorithm, there are three major steps:

Step 1  − Initialized the classifier with the number of nearest neighbors (K).

Step 2  − Fit - Initialized the train and test data.

Step 3 − Predict - Takes the test data and performs the following operations to return a prediction;

* Loops over the test data and calculates the distance between test data and each row of training data using the Euclidean Distance Method.
* Sorts the distance gotten from step 1 in ascending order.
* Next, It gets the position of the first K rows (neighbors).
* Then, it will assign a class to the test point based on most frequent class of these rows.
* Returns the label of the most frequent as the prediction.

## Usage

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn_classifier import KNN

iris = load_iris()
data = iris.data    
target = iris.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=43)

clf = KNN(K = 4) # By Default K = 3
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print('Accuracy:', accuracy_score(y_test, predictions))
```
