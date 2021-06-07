# Feature engineering utility functions
# Author: Anh Tho Le
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

def aggregate_features(df, group_cols, agg_cols, transformations=['mean', 'median', 'min', 'max', 'std']):
    """Create aggregation features: Require a `group` column and a column to do aggregations on.
    
    Args:
        df (DataFrame): Input dataframe
        group_col (list): columns to group 
        agg_col (list): columns to do aggregations
        transformations (list, optional): List of aggregate functions to apply
        Returns
    
    Returns:
        DataFrame: Output dataframe with added features
        list: list of added features
    """
    X = df.copy()
    agg_ft = []
    for c1 in group_cols:
        for c2 in agg_cols:
            for f in transformations:
                X[f'agg_{c1}_{c2}_{f}'] = X.groupby(c1)[c2].transform(f).fillna(0)
                agg_ft.append(f'agg_{c1}_{c2}_{f}')
    return X, agg_ft

def interact_features(df, feature_set):
    """Create interaction features
    
    Args:
        df (DataFrame): Input dataframe
        feature_set1 (list): List of features to get interactions
    
    Returns:
        DataFrame: Output dataframe with added features
        list: list of added features
    """
    X = df.copy()
    inter_cols = []
    for (f1, f2) in combinations(feature_set, 2):
        X[f"inter_{f1}_{f2}_sum"] = X[f1] + X[f2]
        X[f"inter_{f1}_{f2}_sub"] = X[f1] - X[f2]
        X[f"inter_{f1}_{f2}_mul"] = X[f1] * X[f2]
        X[f"inter_{f1}_{f2}_div"] = (X[f1] / X[f2]).fillna(0)
        X[f"inter_{f2}_{f1}_div"] = (X[f2] / X[f1]).fillna(0)
        inter_cols.extend([
            f"inter_{f1}_{f2}_sum",
            f"inter_{f1}_{f2}_sub",
            f"inter_{f1}_{f2}_mul",
            f"inter_{f1}_{f2}_div",
            f"inter_{f2}_{f1}_div",
        ])
    return X, inter_cols

def cat_count_features(df, cat_cols, normalise=False):
    """Get value counts of categorical variables each row
    
    Args:
        df (DataFrame): Input dataframe
        cat_cols (TYPE): Input columns
    
    Returns:
        DataFrame: Output dataframe with added features
        list: list of added features
    """
    X = df.copy()
    l = X.shape[0]
    cat_count_cols = []
    for c in cat_cols:
        d = X[c].value_counts().to_dict()
        X[f'{c}_count'] = X[c].apply(lambda x: d.get(x,0))
        if normalise:
            X[f'{c}_count'] = X[f'{c}_count'] / l
        cat_count_cols.append(f'{c}_count')
    return X, cat_count_cols

def fractional_features(df, float_cols):
    """Get fractional part of float features
    exp: 2.45 --> 0.45
    Args:
        df (DataFrame): Input dataframe
        float_cols (TYPE): Input columns
    
    Returns:
        DataFrame: Output dataframe with added features
        list: list of added features
    """
    X = df.copy()
    fractional_cols = []
    for c in float_cols:
        X[f'{c}_fractional'] = X[c] - np.fix(X[c])
        fractional_cols.append(f'{c}_fractional')

    return X, fractional_cols

def rank_features(df, cols, methods=['average']):
    """Get rank features

    Args:
        df (DataFrame): Input dataframe
        float_cols (TYPE): Input columns
    
    Returns:
        DataFrame: Output dataframe with added features
        list: list of added features
    """
    X = df.copy()
    rank_cols = []
    for c in cols:
        for method in methods:
            X[f'{c}_rank_{method}'] = rankdata(X[c].values, method=method)
            rank_cols.append(f'{c}_rank_{method}')
    return X, rank_cols

def nan_count_features(df):
    X = df.copy()
    X["nan_count"] = X.isna().sum(axis=1).astype(float)
    X["none_count"] = (X == "None").sum(axis=1).astype(float)
    return X

def woe_features(df, iv_df, iv):
    woe_cols = iv[iv['IV'] >= 0.1].VAR_NAME.tolist()
    if len(woe_cols) < 1:
        return df

    X = df.copy()
    for c in woe_cols:
        small_df = iv_df[iv_df['VAR_NAME'] == c][['MAX_VALUE', 'WOE']].rename(columns={'WOE': f'woe_{c}'})
        X = pd.merge(X, small_df, left_on=c, right_on='MAX_VALUE', how='left').drop(columns=['MAX_VALUE'])
    return X

def mean_encoding(X, y, X_test, cols, label="label"):
    """Create mean-encoded features with CV loop to avoid overfitting
    
    Args:
        X (DataFrame): Training data
        y (series): Labels
        X_test (DataFrame): Test data
        cols (list): List of features to encode
        label (str, optional): Label column name
    
    Returns:
        DataFrame: Output train data
        DataFrame: Output test data
    """
    skf = StratifiedKFold(5, shuffle=True, random_state=123)
    df_tr = pd.concat([X, y], axis=1)
    new_df_tr = df_tr.copy()
    # CV loop to avoid overfitting
    for tr_ind, val_ind in skf.split(X, y):
        X_tr, X_val = df_tr.iloc[tr_ind], df_tr.iloc[val_ind]
        for col in cols:
            means = X_val[col].map(X_tr.groupby(col)[label].mean())
            new_df_tr.loc[val_ind, col + "_mean_target"] = means
    prior = df_tr[label].mean()  # global mean
    new_df_tr = new_df_tr.fillna(prior)  # fill NANs with global mean

    X_new = new_df_tr.drop(columns=[label])
    X_test_new = X_test.copy()
    for col in cols:
        mean_mapping = pd.Series(
            X_new[col + "_mean_target"].values, index=X_new[col]
        ).to_dict()
        X_test_new[col + "_mean_target"] = X_test_new[col].map(mean_mapping)
    return X_new, X_test_new