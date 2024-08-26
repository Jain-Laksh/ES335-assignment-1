import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Write the code for Q2 a) and b) below. Show your results.

#Q2 a)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4,stratify=y)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)
y_train = pd.Series(y_train)

print("\nQuestion 2: Part A")
print("X_train.shape:",X_train.shape)
print("X_test.shape:",X_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  #max-depth set to 5
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    print("\nCriteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y_test))
    for cls in y_test.unique():
        print("Precision: ", precision(y_hat, y_test, cls))
        print("Recall: ", recall(y_hat, y_test, cls))

#Q2 b)

# Using 5 fold Cross-Validation

print("\nQuestion 2: Part B: K cross validation")
k = 5   # Define the number of folds (k)

accuracies = [] 
fold_size = len(X) // k

for criteria in ["information_gain", "gini_index"]:
    print(f"\nCriteria : {criteria}")
    for i in range(k):
        # Split the data into training and test sets
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_set = X[test_start:test_end]
        test_labels = y[test_start:test_end]
        
        training_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
        training_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)
        
        # Train the model
        dt_classifier = DecisionTree(criterion= criteria)
        dt_classifier.fit(pd.DataFrame(training_set), pd.Series(training_labels))
        
        # Make predictions on the validation set
        fold_predictions = dt_classifier.predict(pd.DataFrame(test_set))
        
        # Calculate the accuracy of the fold
        fold_accuracy = np.mean(fold_predictions == test_labels)
        
        # Store the predictions and accuracy of the fold
        accuracies.append(fold_accuracy)

    # Print the predictions and accuracies of each fold
    for i in range(k):
        print("Fold {}: Accuracy: {:.4f}".format(i+1, accuracies[i]))
    print("Mean Accuracy:",np.mean(accuracies))

# Using nested cross-validation to find the optimum depth of the tree

print("\nQuestion 2: Part B: Using nested cross-validation to find the optimum depth of the tree")

max_depth_values = [1,2,3,4,5,6,7,8,9,10]
criteria_values = ['information_gain', 'gini_index']

best_accuracy = 0
best_params = {}

X_train_val,X_test,y_train_val,y_test = train_test_split(X,y,test_size=0.3,random_state=4)
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=4)
X_train_val = pd.DataFrame(X_train_val)
y_train_val = pd.Series(y_train_val)
X_val = pd.DataFrame(X_val)
y_val = pd.Series(y_val)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)
y_train = pd.Series(y_train)

for max_depth in max_depth_values:
    for criterion in criteria_values:
        # Define the Decision Tree Classifier
        dt_classifier = DecisionTree(max_depth=max_depth,criterion=criterion)
        dt_classifier.fit(X_train, y_train)
        
        # Evaluate on the validation set
        y_hat_val = dt_classifier.predict(X_val)
        val_accuracy = accuracy(y_hat_val, y_val)
        
        # Check if this combination gives a better accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = {'max_depth': max_depth,'criterion': criterion}

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)
print("Best Validation Accuracy:", best_accuracy)

# Train the model with the best hyperparameters
best_dt_classifier = DecisionTree(criterion=best_params["criterion"] , max_depth=best_params["max_depth"])
best_dt_classifier.fit(X_train_val, y_train_val)

# Evaluate on the test set
y_hat_test = best_dt_classifier.predict(X_test)
test_accuracy = accuracy(y_hat_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")