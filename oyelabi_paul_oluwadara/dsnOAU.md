# A Light Implementation of K neares neighbors.

A light implementation of K nearest neighbors algorithm.
This implementation only supports euclidean and manhattan
distance metrics. It also provides the probability of
prediction. I leveraged on Numpy and python built-in 
statistics package.
    
It provides common interface as the sklearn kNN algorithm.
Below are runs of the sklearn and my implementation.
       
       parameters:
           k: the numbers of neighbors.
           method: euclidean/manhattan.
           mode: classification/regression.
           
        
        fit method:
            The KNN algorithm doesn't really learn, this method only
            captures the training dataset with which test samples will 
            be compared.
            
        predict method:
            The prediction is made by comparing the distance of datapoints
            in the training set to the the test sample. This method calls 
            the util method which calculates the distance metric and then
            assign the class with majority vote(with highest number of
            occurence) to the test sample.
          
        predict_proba method:
            This method works like the predict method, it calls the proba_util
            method which performs almost same functionality as the util method
            but it calculates the probability of the test belonging to what
            predict mwthods performs.
            
        The argpartition method used in the distance metrics works by partitioning
        the array into two, with the k least numbers on the left and others on the
        right and then returns the indices of the k-least numbers, which is then 
        used to index the target variable to get the corresponding labels.
    
`clf_1 --> sklearn KNN classifier algorithm`

`clf_2 --> my algorithm`

`regr_1 --> sklearn KNN regressor algorithm`

`regr_2 -- my algorithm`
    
The implementation is found in knn.py file.
    


```python
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from timeit import timeit
from knn import KNN
```

# CLASSIFICATION - iris flower classification


```python
#load classification data
data, target = load_iris(True)
data.shape, target.shape
```




    ((150, 4), (150,))




```python
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0, shuffle=True)
```

### sklearn knn classifier


```python
clf_1 = KNeighborsClassifier(n_neighbors=3)
```


```python
clf_1.fit(x_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                         weights='uniform')




```python
y1_pred = clf_1.predict(x_test)
```


```python
print("sklearn KNN classifier accuracy score: {:.2f}%".format(accuracy_score(y_test, y1_pred)*100))
```

    sklearn KNN classifier accuracy score: 97.37%
    


```python
clf_1.predict_proba(x_test)
```




    array([[0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.66666667, 0.33333333],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.33333333, 0.66666667],
           [0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ]])



### My KNN classifier


```python
clf_2 = KNN() #defaults to 3 nearest neighbors
```


```python
clf_2.fit(x_train, y_train)
```


```python
y2_pred = clf_2.predict(x_test)
```


```python
print("My KNN classifier accuracy score: {:.2f}%".format(accuracy_score(y_test, y2_pred)*100))
```

    My KNN classifier accuracy score: 97.37%
    


```python
clf_2.predict_proba(x_test)
```




    array([[0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.66666667, 0.33333333],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.33333333, 0.66666667],
           [0.        , 0.        , 1.        ],
           [0.        , 1.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        ]])



# CLASSIFICATION - breast cancer


```python
breast_data, breast_target = load_breast_cancer(True)
breast_data.shape, breast_target.shape
```




    ((569, 30), (569,))




```python
x2_train, x2_test, y2_train, y2_test = train_test_split(
            breast_data, breast_target, test_size=0.25, random_state=0, shuffle=True
        )
```

### sklearn knn classifier


```python
clf_1 = KNeighborsClassifier(n_neighbors=3)
```


```python
clf_1.fit(x2_train, y2_train);
```


```python
y1_pred = clf_1.predict(x2_test)
```


```python
print("sklearn KNN classifier accuracy score: {:.2f}%".format(accuracy_score(y2_test, y1_pred)*100))
```

    sklearn KNN classifier accuracy score: 92.31%
    


```python
clf_1.predict_proba(x2_test)
```




    array([[0.66666667, 0.33333333],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [0.66666667, 0.33333333],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.66666667, 0.33333333],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.33333333, 0.66666667],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.33333333, 0.66666667],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.66666667, 0.33333333],
           [1.        , 0.        ],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ]])



### My KNN classifier


```python
clf_2 = KNN() #defaults to 3 nearest neighbors
```


```python
clf_2.fit(x2_train, y2_train)
```


```python
y2_pred = clf_2.predict(x2_test)
```


```python
print("My KNN classifier accuracy score: {:.2f}%".format(accuracy_score(y2_test, y2_pred)*100))
```

    My KNN classifier accuracy score: 92.31%
    


```python
clf_2.predict_proba(x2_test)
```




    array([[0.66666667, 0.33333333],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [0.66666667, 0.33333333],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.66666667, 0.33333333],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.66666667, 0.33333333],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.33333333, 0.66666667],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.33333333, 0.66666667],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.66666667, 0.33333333],
           [1.        , 0.        ],
           [0.66666667, 0.33333333],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.33333333, 0.66666667],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ]])



# REGRESSION - diabetes dataset


```python
rgr_data, rgr_target = load_diabetes(True)
rgr_data.shape, rgr_target.shape
```




    ((442, 10), (442,))




```python
rx_train, rx_test, ry_train, ry_test = train_test_split(rgr_data, rgr_target, test_size=0.25, random_state=0, shuffle=True)
```

### sklearn KNN regressor


```python
regr_1 = KNeighborsRegressor(n_neighbors=3)
regr_1.fit(rx_train, ry_train);
```


```python
ry_pred = regr_1.predict(rx_test)
```


```python
print("sklearn KNN regressor RMSE: {:.2f}".format(mean_squared_error(ry_test, ry_pred)))
```

    sklearn KNN regressor RMSE: 4232.01
    

### My KNN regressor


```python
regr_2 = KNN(k=3, mode="regression")
regr_2.fit(rx_train, ry_train);
```


```python
ry2_pred = regr_2.predict(rx_test)
```


```python
print("My KNN regresRMSE: {:.2f}".format(mean_squared_error(ry_test, ry2_pred)))
```

    My KNN regressor RMSE: 4232.01
    


```python

```
