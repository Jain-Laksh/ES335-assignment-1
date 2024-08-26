"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

# import os
# curr_dir = os.getcwd()
# print(curr_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class TreeNode:

    def __init__(self, feature=None, threshold=None, left=None, right=None,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

        # for leaf node
        self.value = value


class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="gini_index", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def grow_tree(self,X,y, depth=0):

        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y)) 

        # check the stopping criteria

        if (depth>= self.max_depth or n_labels<=1 or opt_split_attribute(X,y,self.criterion,features = X.columns.values.tolist())[0] is None):
            leaf_value = leaf_node_value(y)
            return TreeNode(value=leaf_value)

        # find the best split
 
        best_split = best_split_dict(X,y,self.criterion)
 
        # create child nodes 

        left_child = self.grow_tree(best_split["X_left"] , best_split["y_left"] , depth=depth+1)
        right_child = self.grow_tree(best_split["X_right"] , best_split["y_right"] , depth=depth+1)
        
        return TreeNode(best_split["feature"] , best_split["threshold"] , left_child , right_child)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
 
        # Function to train and construct the decision tree

        X = one_hot_encoding(X)
        self.root = self.grow_tree(X,y)  


    def predict(self, X: pd.DataFrame) -> pd.Series:

        # Funtion to run the decision tree on test inputs
        predictions = []
        X = one_hot_encoding(X)
        for index, row in X.iterrows():
            predictions.append(self.traverse_tree(row, self.root))
        return pd.Series(predictions)     

    def traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        def recurse(node, indent=""):
            if node.value is not None:
                print(f"{indent}Class {node.value}")
                return
            print(f"{indent}?(X{node.feature} > {node.threshold})")
            print(f"{indent}Y: ", end="")
            recurse(node.left, indent + "    ")
            print(f"{indent}N: ", end="")
            recurse(node.right, indent + "    ")

        recurse(self.root)
