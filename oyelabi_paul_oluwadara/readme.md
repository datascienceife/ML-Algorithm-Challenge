# A Light Implementation of K neares neighbors.

A light implementation of K nearest neighbors algorithm.
This implementation only supports euclidean and manhattan
distance metrics. It also provides the probability of
prediction. I leveraged on Numpy and python built-in 
statistics package.
    
It provides common interface as the sklearn kNN algorithm.
Below are runs of the sklearn and my implementation.

The KNN algorithm works by calculating how close datapoints are
to one another by calculating the distance between two datapoints
using distance metrics like euclidean distance and then select K
closest datapoints. It then assign the class with majority vote in
the K selected datapoints to the new sample the algorithm is trying
to classify. And in terms of regression, it calculates the mean of
the K selected datapoints to the new sample.
       
       parameters:
           k: the numbers of neighbors.
           method: euclidean/manhattan.
           mode: classification/regression.
           
        
        fit method:
            Since the KNN algorithm doesn't really learn, this method only
            captures the predictors (X) and target (Y) with which test samples
            will be compared.
            
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
            
        euclidean method:
            This method calculates the euclidean distance between datapoints using
            the numpy linear algebra method. I used the numpy argpartition method
            to sort the calculated distance and then it returns the indices of the
            K closest (smallest value) distances.
        
        manhattan method:
            This method calculates the manhattan distance between datapoints using
            the numpy linear algebra method. I used the numpy argpartition method
            to sort the calculated distance and then it returns the indices of the
            K closest (smallest value) distances.
            
        util method:
             This method calls either euclidean or manhattan method according to the
             method parameter passed to the class constructor. It gets the indices of
             the K closest datapoints which it then uses to index the target varible
             to get the corresponding class label. It then calculates the majority
             vote using the python built-in statistics mode method. If the mode parameter
             is set to `regression` this method returns the mean of the K nearest datapoints.
             
             `I could have assigned class randomly whenever there is a tie in the majority
              vote, but instead, I raised a warning and then advise the data scientist to
              consider seecting the K value with respect to the number of classes (i.e. an
              even value for K whenever the number of class in target varible is odd, and an
              odd value for K whenever that number of class in target variable is even) in the
              target varible as this is a best practice to ensure good results.`
              
         prob_util method:
             This method is only available during classification mode. This method works exactly
             like the util method but it returns the probability of a test sample belonging to 
             the predicted class. It acheive this by dividing the total number of occurence
             of each class in the K nearest datapoints by K.
            
        The argpartition method used in the distance metrics works by partitioning
        the array into two, with the k least numbers on the left and others on the
        right and then returns the indices of the k-least numbers, which is then 
        used to index the target variable to get the corresponding labels.
    
### Some tests are found in [dsnOAU.md](https://github.com/Yodeman/ML-Algorithm-Challenge/blob/main/oyelabi_paul_oluwadara/dsnOAU.md)
