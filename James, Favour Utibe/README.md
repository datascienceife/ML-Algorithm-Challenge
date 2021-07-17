# ML-Algorithm-Challenge

KNN stands for K-Nearest Neighbors. The k-nearest neighbors algorithm uses a very simple approach to perform classification. When tested with a new example, it looks through the training data and finds the k training examples that are closest to the new example. It then assigns the most common class label (among those k-training examples) to the test example.
k in the kNN algorithm represents the number of nearest neighbor points which are voting for the new test data’s class. 

If k=1, then test examples are given the same label as the closest example in the training set. If k=3, the labels of the three closest classes are checked and the most common (i.e., occurring at least twice) label is assigned, and so on for larger ks.

To calculate the distance between the points, different types of distance metrics can be used such as euclidean distance, cosine distance, and so on, though the euclidean distance is most widely used because it still functions better on most datasets and that is what is used in the sklearn module.

How the algorithm functions

In this algorithm, I made use of the euclidean distance. 

Euclidean distance is just a straight-line distance between two data points in Euclidean space[1]. It can be calculated as follows:

d(x,y) = ((x1 - y1)² + (x2 - y2)² + ... + (xn - yn)²)½

But this way of calculating distance accounts for only 2-dimensional points, in the case of 3-dimensional points, it won't function well. In my code, I made use of the numpy method of calculating distances and this also accounts for 3-dimensional points.

Algorithm Implementation

 k: value for k
train_set: entire list with values for training the algorithm
test_set: entire list with values for testing the algorithm

Steps to follow:

1. Calculate Euclidean distance between the test_instanceand each row of the train_set
2. Sort the distances by distance value, from lowest to highest
3. Keep the distance of the smallest ones
4. Get values of a target variable for k train_set rows with the smallest distance
5. Whichever target variable class has the majority, wins

This code is made up of 2 main functions:

1. getPredictions: This function is written to calculate the euclidean distance between the data points, get the nearest neighbors according to which is nearest to the point and select the most common class of the nearest neighbors.

2. evaluate: This function is written to test the algorithm using the train and test sets.





