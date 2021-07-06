# ML-Algorithm-Challenge

| ![community logo](oau.png) | AI+ OAU Algorithm Challenge |
| -------------------------- | --------------------------- |

NAME : Knn-classifier 

Description : Creating a Knn-classifier from scratch 

The code is divided into FOUR main functions:
1. The distance function :
    This function takes two parameters as input. Two points and then we use the formula to calculate the distance between two points.
	Assume the points are (2, 2) and (3, 3)
	first, the difference is calculated (3 - 2) and (3 - 2) and the squares of both are calculated equalling 1 each
	second, we sum the squared differences which equals 2 
	third, we take the square root of the sum
	this can be done for any number of points say 3, 4, 5 e.t.c but both points to be subtracted must be of the same length
	This is achieved using the numpy sqrt(square root)  and sum functions
	
2. The majority votes function:
	This function is used to calculate the occurence that happens the most from a list of occurences
	We are given a list of different things then start a for loop, if the current occurnece has been observed once, we increase its value by 1 else, we add the 
	occurence to the dictionary used to keep track.
	After the for loop ends, we get the max value of the occurences and if more than one happens to have the max value, one is chosen randomly.

3. find nearest neighbours fuction: 
	This function is to find the the k-nearest neighbour. It takes three parameters p, points and k.
	* p is the point which we want to determine its neighbours
	* points is a list of points that surrounds where the point p can lie/ fall in and this is where the closest k-neighbours are gotten. 
	* k is the number of neighbours we would like to consider. The default value is set to 5
	A for loop is started and we call the distance function we created to calculate the distance betwwen the point(p) and every point in points param. and the results 
	are kept in a list. Argsort fuction of numpy is then used to sort the list in order of ascending values and we then choose a slice of up to K.
	since the input parameters are numpy arrays, the indexes rather than the values of the closest neighbours are returned so as to be able to use the indexes to 
	get the outcome values
4. The KNN_predict fuction:
	This function performs the prediction of a new point and takes 4 parameters p, points, outcomes, k
	* p is the point which we want to predict
	* points is the list of points that surrounds where the point p can lie/ fall in, this is where the closest neighbours of k are gotten.
	* outcomes is the outcome of the each point in points and from which we predict the outcome of point p
	* k is the number of neighbours we would like to consider. The default value is set to 5
	we call the find nearest neighbours fuction on a point p and set the indexes gotten to a list called ind
	we then call the majority vote function on the list of a sliced list containing just the outcomes of the indexes of the nearest neighnours
	
In most real world cases p will not be a single point to predict but rather a list of points to predict therefore the code can be run in 2 methodss:
	First method: 
		Using contemporary method by creating a list to keep the values of our predictions,
		craete a for loop to iterate through all the points we want to predict and for each point prdict the outcome and add the result to the created list
		code snippet:
		    prediction_points = [points that we want to predict]
			predictors = [points for which we know the outcomes]
			outcomes = [outcomes for each point in point]
		    predicted_outcomes = []
			for p in prediction_points:
			    prediction = knn_predict(p, predictors, outcomes, k = 5)
				predicted_outcomes.append(prediction)
			
	Second method :
		Using list comprehension which is easier
		code snippet:
			predicted_outcomes = [knn_predict(p, predictors, outcomes, k = 5) for p in prediction_points]

To test the code we use the sklearn module to get some data set and use our just created classifier to predict the values and compare the results with that of the
Sklearn model and the results are very similar which shows a little amount of accuracy in our model

NOTE : All inputs must be numerical and is advisable to be a numpy array