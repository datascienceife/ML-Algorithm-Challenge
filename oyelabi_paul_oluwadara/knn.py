import numpy as np
import statistics as st

class KNN():
    """
    A light implementation of K nearest neighbors algorithm.
    This implementation only supports euclidean and manhattan
    distance metrics. It also provides the probability of
    prediction.

    parameters:
       k: the numbers of neighbors.
       method: euclidean/manhattan.
       mode: classification/regression.
    """

    def __init__(self, k=3, mode="classification", method="euclidean"):
        self.k = k
        self.mode = mode
        self.x = None
        self.y = None
        self.class_ = None
        self.method = method

        assert self.mode in ("classification", "regression"), "Unsupported mode."
        assert self.method in ("euclidean", "manhattan"), "Unsupported method."

    def fit(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            self.x = x
            self.y = y
        else:
            self.x = np.array(x)
            self.y = np.array(y)
        self.class_ = np.unique(self.y)

    def predict(self, x):
        assert isinstance(self.x, np.ndarray) and isinstance(self.y, np.ndarray),\
        "You need to train before predicting."
        if (x.ndim < 2): raise Exception("Input not in right shape.")
        return np.apply_along_axis(self.util, 1, x)

    def predict_proba(self, x):
        assert self.mode=="classification", "Method availabel only for classification."
        assert isinstance(self.x, np.ndarray) and isinstance(self.y, np.ndarray),\
        "You need to train before predicting."
        return np.apply_along_axis(self.prob_util, 1, x)

    def euclidean(self, x):
        return np.argpartition(
            np.linalg.norm(self.x-x, axis=1), self.k)[:self.k]
        #np.sqrt(np.sum(np.square(self.x-x), 1))

    def manhattan(self, x):
        return np.argpartition(
            np.linalg.norm(self.x-x, ord=1, axis=1), self.k)[:self.k]
        #np.sqrt(np.sum(np.absolute(self.x-x), 1))

    def prob_util(self, x):
        idx = self.euclidean(x) if self.method=="euclidean" else self.manhattan(x)
        cls_ = self.y[idx]
        #pred = st.mode(cls_)
        out = []
        unique, count = np.unique(cls_, return_counts=True)
        for i in self.class_:
            try:
                out.append(count[np.where(unique==i)][0]/len(cls_))
            except IndexError:
                out.append(0)
        return out

    def util(self, x):
        idx = self.euclidean(x) if self.method=="euclidean" else self.manhattan(x)
        cls_ = self.y[idx]
        if self.mode=="classification":
            try:
                return st.mode(cls_)
            except:
                raise Exception(
                    "no unique mode; found 2 equally common values"
                    "You should consider the value of k with respect"
                    "to the number of classes in your target variable."
                    ) from None
                
        elif self.mode=="regression":
            return st.mean(cls_)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    data, target = load_iris(True)
    x_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0, shuffle=True)
    clf_2 = KNN(method="manhattan")
    clf_2.fit(x_train, y_train)
    print(clf_2.class_)
    y = clf_2.predict(X_test)
    print("My KNN classifier accuracy score: {:.2f}%".format(accuracy_score(y_test, y)*100))
    print(clf_2.predict_proba(X_test))
