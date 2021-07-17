import numpy as np
import pandas as pd
import warnings
from collections import Counter
import random
from sklearn.model_selection import train_test_split

class k_nearest_neighbors(object):
        def __init__(self, k):
                self.k = k
                
        def getPrediction(self, df, predict):
            if len(df)>=self.k:
                warnings.warn('k is set to a value less than total voting groups! ')

        ##getting the distance between the former points and the new points
            distances = []
            for group in df:
                for features in df[group]:
                    EuclideanDistance = np.linalg.norm(np.array(features) - np.array(predict))
                    distances.append([EuclideanDistance, group])

        ##getting the nearest neighbors of the new points based on the closest distance
            classVotes = [i[1] for i in sorted(distances)[:self.k]]
            kNeighbors = Counter(classVotes).most_common(1)[0][0]
            return kNeighbors
        
        def evaluate(self, train_set, test_set):
                correct = 0
                total = 0
                for group in test_set:
                    for data in test_set[group]:
                        vote = self.getPrediction(train_set, data)
                        if group == vote:
                            correct += 1
                        total += 1
                    
                print('Accuracy:',correct/total)
                return correct/total
                

                

