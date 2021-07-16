import numpy as np
import operator

# calculate euclidean distance
def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN():
    
    def __init__(self, K = 3):
        if(K < 1):
            raise Exception('K must be grater than 0')
        self.K = K

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.y_train = y_train

    def predict(self, X_test):
        
        # list to store all our predictions
        predictions = []
        
        # loop over all observations
        for i in range(len(X_test)):            
            
            # calculate the distance between the test point and all other points in the training set
            dist = np.array([euc_dist(X_test[i], x) for x in self.X_train])
            
            # sort the distances and return the positions of the first K neighbors
            dist_sorted = dist.argsort()[:self.K]
            
            neighbor_votes = {}

            # for each neighbor find the class and return the most voted.
            for d in dist_sorted:
                if self.y_train[d] in neighbor_votes:
                    neighbor_votes[self.y_train[d]] += 1
                else:
                    neighbor_votes[self.y_train[d]] = 1
            
            # get the most common class label 
            sorted_neighbors = sorted(neighbor_votes.items(), key=operator.itemgetter(1), reverse=True)
            
            predictions.append(sorted_neighbors[0][0])
        return predictions