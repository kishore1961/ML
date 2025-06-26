# p(y) = prior probability => frequency of each class
# P(x|y) = class conditional probability => model with gaussian

import numpy as np

class NaiveBayes:
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate the mean, var and prior for each class
        self.mean = np.zeros((n_classes,n_features) , dtype = np.float64)
        self.var = np.zeros((n_classes,n_features) , dtype = np.float64)
        self.proors = np.zeros(n_classes , dtype = np.float64)

        for idx , c in enumerate(self._classes):
            


    def predict(self, X,y):
        pass
         