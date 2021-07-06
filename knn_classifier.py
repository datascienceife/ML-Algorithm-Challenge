import numpy as np
import random

def distance(p1, p2):
    '''find the distance between two points'''
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def majority_vote(votes):
    ''' from a list/sequence of votes, count the frequency of each vote, then return the vote with highest value
    if there's a tie, one vote is selected at random'''
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)
    return random.choice(winners)

def find_nearest_neighbours(p, points, k=5):
    '''find the k nearest neighbours and return their corresponding indices'''
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    '''use the find nearest neighbors function to find indexes of nearest points to p,
    then use majority votes function to decide the outcome of p based on outcome at indexes'''
    ind = find_nearest_neighbours(p, points, k)
    return majority_vote(outcomes[ind])

# Testing the knn_classifier using the iris dataset

def accuracy(predictions, outcomes):
    correct = 0
    wrong = 0
    for i in range(len(predictions)):
        if predictions[i]  == outcomes[i]:
            correct += 1
        else:
            wrong += 1
    return correct / len(predictions)


from sklearn import datasets
iris = datasets.load_iris()
predictors = iris['data']
outcomes = iris.target

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(predictors, outcomes)
sk_predict = knn.predict(predictors)

my_predict= np.array([knn_predict(p, predictors, outcomes) for p in predictors])
print(accuracy(my_predict, outcomes))
print(accuracy(outcomes, outcomes))
print(accuracy(outcomes, sk_predict))