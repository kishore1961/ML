import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature = None , threshold = None , left = None, right = None, info_gain = None, value = None):
        
        # for decision node
        
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self,min_samples_split = 2, max_depth = 1000, n_features = None):
        
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features


    def fit(self , X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth =0):

        # n_features = 30

        n_samples , n_features = X.shape
        n_labels = len(np.unique(y))
        # print("length " , len(y))
        # print("n_labels" , n_labels)
        # check stopping criteria

        if ( depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            # pass the value to the code if there is only one label then we can directly pass the value or else we need to fins the value
            # print("most_common label : " , y)
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx = np.random.choice(n_features , self.n_features, replace = False)

        # find the best split
        best_feature, best_threshold  = self._best_split(X,y,feat_idx)

        # create child nodes

        left_idxs , right_idxs = self._split(X[:,best_feature] , best_threshold)
        left = self._grow_tree(X[left_idxs,:] , y[left_idxs] , depth+1)
        right = self._grow_tree(X[right_idxs,:] , y[right_idxs] , depth+1)

        return Node(best_feature, best_threshold , left, right)



    # best split       

    def _best_split(self,X,y,feature_idx):
        
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feature_idx:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

        for thr in thresholds:
            # calculate information gain

            gain = self._information_gain(y,X_column , thr)

            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
                split_threshold = thr
        return split_idx , split_threshold

    def _information_gain(self, y,X,thr):
        
        # parent entropy

        parent_entropy = self._entropy(y) 
        # create children
        left_idx,right_idx = self._split(X, thr)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # calculate the weight of entropy of children

        n = len(y)
        n_l , n_r = len(left_idx) , len(right_idx)
        e_l , e_r = self._entropy(y[left_idx]) , self._entropy(y[right_idx])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        # calculate the IG

        information_gain = parent_entropy - child_entropy
        return information_gain
        

    def _split(self, X,threshold):
        left_idxs = np.argwhere(X <= threshold).flatten()
        right_idxs = np.argwhere(X > threshold).flatten()

        return left_idxs , right_idxs


    def _entropy(self, y):
        # print("entropy...." , y)
        hist = np.bincount(y)
        ps = hist/len(y)

        return -np.sum([p*np.log(p) for p in ps if p>0])

    # find the most common value
    def _most_common_label(self,y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]       
        
    def predict (self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self,X, node):
        if node.is_leaf_node():
            return node.value
        
        if X[node.feature] <= node.threshold:
            return self._traverse_tree(X,node.left)
        return self._traverse_tree(X,node.right)
        
        
        
         