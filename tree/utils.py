import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_float_dtype

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:

    one_hot_encoded_df = pd.get_dummies(X)
    return one_hot_encoded_df*1                 # multiplying dataframe with 1 converts all boolean values to int type and does not affect str or float values
    
def check_ifreal(y: pd.Series) -> bool:

    if is_string_dtype(y) :
        return False
    
    elif is_numeric_dtype(y):
        if is_float_dtype(y):
            return True
        else:
            return False

    return False

def entropy(y: pd.Series) -> float:

    hist = np.bincount(y)
    pb = hist / len(y)
    return -np.sum([p * np.log2(p) for p in pb if p>0])

def gini_index(Y: pd.Series) -> float:

    hist = np.bincount(Y)
    pb = hist / len(Y)
    return  1 - np.sum([p*p for p in pb])

def gini_gain(Y: pd.Series, attr: pd.Series, threshold) -> float:

    #Calculating gain in gini index for discrete output

    if not check_ifreal(Y):
        initial_gini = gini_index(Y)
        in_out_df = pd.DataFrame()
        in_out_df['feature'] = attr
        in_out_df['output'] = Y
        left_split = in_out_df[in_out_df['feature'] <= threshold]['output']
        right_split = in_out_df[in_out_df['feature'] > threshold]['output']
        gini_left = gini_index(left_split)
        gini_right = gini_index(right_split)
        gini_gain = initial_gini - len(left_split)/len(Y) * gini_left - len(right_split)/len(Y) * gini_right
        return gini_gain

    #Calculating information gain using mse for real output

    else:
        initial_mse = np.mean((np.mean(Y) - Y)**2)
        in_out_df = pd.DataFrame()
        in_out_df['feature'] = attr
        in_out_df['output'] = Y
        left_split = in_out_df[in_out_df['feature'] <= threshold]['output']
        right_split = in_out_df[in_out_df['feature'] > threshold]['output']
        mse_left = np.mean((np.mean(left_split) - left_split)**2)
        mse_right = np.mean((np.mean(right_split) - right_split)**2)
        information_gain = initial_mse - len(left_split)/len(Y) * mse_left - len(right_split)/len(Y) * mse_right
        return information_gain

def information_gain(Y: pd.Series, attr: pd.Series,threshold) -> float:

    #Calculating information gain using entropy for discrete output

    if not check_ifreal(Y):
        initial_entropy = entropy(Y)
        in_out_df = pd.DataFrame()
        in_out_df['feature'] = attr
        in_out_df['output'] = Y
        left_split = in_out_df[in_out_df['feature'] <= threshold]['output']
        right_split = in_out_df[in_out_df['feature'] > threshold]['output']
        entropy_left = entropy(left_split)
        entropy_right = entropy(right_split)
        information_gain = initial_entropy - len(left_split)/len(Y) * entropy_left - len(right_split)/len(Y) * entropy_right

    #Calculating information gain using mse for real output
    
    else:
        initial_mse = np.mean((np.mean(Y) - Y)**2)
        in_out_df = pd.DataFrame()
        in_out_df['feature'] = attr
        in_out_df['output'] = Y
        left_split = in_out_df[in_out_df['feature'] <= threshold]['output']
        right_split = in_out_df[in_out_df['feature'] > threshold]['output']
        mse_left = np.mean((np.mean(left_split) - left_split)**2)
        mse_right = np.mean((np.mean(right_split) - right_split)**2)
        # print(len(Y))
        information_gain = initial_mse - len(left_split)/len(Y) * mse_left - len(right_split)/len(Y) * mse_right

    return information_gain


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):

    # Function to find the optimal attribute to split about.

    if criterion == 'information_gain':
        best_info_gain = -1
        best_feature = None
        best_thres = None

        for feature in features:
            threshold = split_threshold(y,X[feature],criterion)
            if threshold is not None:
                info_gain = information_gain(y, X[feature],threshold)
                # print(info_gain)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_thres = threshold

    elif criterion == 'gini_index':
        best_gini_gain = -1
        best_feature = None
        best_thres = None

        for feature in features:
            threshold = split_threshold(y,X[feature],criterion)
            if threshold is not None:
                _gini_gain = gini_gain(y, X[feature],threshold)
                if _gini_gain > best_gini_gain:
                    best_gini_gain = _gini_gain
                    best_feature = feature
                    best_thres = threshold

    # print(best_feature)
    return [best_feature , best_thres]

def split_threshold(Y: pd.Series, attr: pd.Series,criterion):

    unique_vals = np.unique(attr)
    thresholds = []
    for i in range(len(unique_vals)-1):
        thresholds.append(np.mean([unique_vals[i],unique_vals[i+1]]))

    # print(unique_vals , thresholds)
    if criterion == "information_gain":
        best_info_gain = -1
        best_threshold = None
        for threshold in thresholds:
            # print("threshold , info gain calculated:",threshold , information_gain(Y,attr,threshold))
            if information_gain(Y,attr,threshold) > best_info_gain:
                best_info_gain = information_gain(Y,attr,threshold)
                # print("info gain:",best_info_gain)
                best_threshold = threshold
        
    elif criterion == "gini_index":
        best_gini_gain = -1
        best_threshold = None
        for threshold in thresholds:
            if gini_gain(Y,attr,threshold) > best_gini_gain:
                best_gini_gain = gini_gain(Y,attr,threshold)
                best_threshold = threshold

    # print(best_threshold, criterion, len(Y))
    return best_threshold

def split_data(X: pd.DataFrame, y: pd.Series, attribute, threshold):

    # Split the data based on a particular value of a particular attribute. 

    in_out_df = X
    in_out_df["output"] = y
    X_left= in_out_df[in_out_df[attribute] <= threshold].drop(['output'], axis=1)
    y_left= in_out_df[in_out_df[attribute] <= threshold]['output']
    X_right = in_out_df[in_out_df[attribute] > threshold].drop(['output'], axis=1)
    y_right = in_out_df[in_out_df[attribute] > threshold]['output']

    return X_left,y_left,X_right,y_right

def leaf_node_value(y):
    if not check_ifreal(y):
        frequency ={}
        for item in y:
            if item not in frequency:
                frequency[item]=1
            else:
                frequency[item]+=1
        
        mode = -1
        most_common_value = None
        for value,count in frequency.items():
            if count>mode:
                mode = count
                most_common_value = value
        
        return most_common_value
    
    else:
        return np.mean(y)
            
def best_split_dict(X,y,criterion):

    best_split = {}
    features = X.columns.values.tolist()
    feat , thres =  opt_split_attribute(X, y, criterion, features)
    best_split["feature"] = feat
    best_split["threshold"] = thres
    X_left,y_left,X_right,y_right = split_data(X, y, best_split["feature"], best_split["threshold"])
    best_split["X_left"] = X_left
    best_split["X_right"] = X_right
    best_split["y_left"] = y_left
    best_split["y_right"] = y_right

    return best_split

if __name__ == "__main__":
    series = pd.Series(['cat', 'dog', 'mouse', 'cat'])
    test = pd.DataFrame()
    test['Animals'] = ['cat', 'dog', 'mouse', 'cat']
    test['weights'] = [54.2,48.0,10.1,50.5]
    print(test)
    print(check_ifreal(series))
    print(one_hot_encoding(series))
    print(one_hot_encoding(test))
    # N = 30
    # P = 5
    # X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
    # y = pd.Series(np.random.randn(N))
    # print(X)
    # temp_df = one_hot_encoding(X)
    # print(temp_df)
    # print(X.columns.values.tolist())
    # print(temp_df.columns.values.tolist())