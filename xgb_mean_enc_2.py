# XGBoost: 3 new features (n_missing, bin_sum, bin_diff), no dropping of any feature. Use average of 5-fold CV, early stopping 100 rounds
# add new feature: average age

import numpy as np
import pandas as pd
from utils.metrics import gini
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest, BaggingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, cross_validate
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_recall_fscore_support, precision_score, recall_score, make_scorer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
from utils.feature_engineer import *
from utils.preprocessing import *
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
import xgboost as xgb
from xgboost import plot_importance
# from imblearn.over_sampling import SMOTE
import optuna
from tqdm import tqdm
import logging, warnings, sys
import random as rn
import time
# Disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Log configuration
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Funcitons from olivier's kernel
# https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -gini(labels, preds)
    return [('gini', gini_score)]

def main():
    logging.info("Loading data...")
    train_df = pd.read_csv("data/train.csv", low_memory=False)
    test_df = pd.read_csv("data/test.csv", low_memory=False)
    train_df.columns = map(str.lower, train_df.columns)
    test_df.columns = map(str.lower, test_df.columns)

    # Load folds
    folds = []
    for i in range(5):
        trn_ind = np.load(f'folds/fold_{i}_train.npy')
        test_ind = np.load(f'folds/fold_{i}_test.npy')
        folds.append((trn_ind, test_ind))

    # Cleaning
    df = pd.concat([train_df, test_df], ignore_index=True)
    logging.info("Cleaning...")
    roman_conversion = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    cat_cols = [
        "province",
        "district",
        "macv",
        "field_7",
        "field_8",
        "field_9",
        "field_10",
        "field_12",
        "field_13",
        "field_17",
        "field_24",
        "field_35",
        "field_39",
        "field_40",
        "field_42",
        "field_43",
        "field_44",
    ]

    ind_cols = [
        "field_1",
        "field_4",
        "field_5",
        "field_6",
        "field_11",
        "field_14",
        "field_15",
        "field_16",
        "field_21",
        "field_32",
        "field_33",
        "field_34",
        "field_41",
        "field_45",
        "field_46",
    ]

    bin_cols = [
        "field_2",
        "field_18",
        "field_19",
        "field_20",
        "field_23",
        "field_25",
        "field_26",
        "field_27",
        "field_28",
        "field_29",
        "field_30",
        "field_31",
        "field_36",
        "field_37",
        "field_38",
        "field_47",
        "field_48",
        "field_49",
    ]

    num_cols = [
        "age_source1",
        "field_3",
        "field_22",
        "field_53",
        "field_54",
        "field_55",
        "age_source2",
        "field_50",
        "field_51",
        "field_52",
        "field_56",
        "field_57",
    ]

    # New features: Counts of NaNs and Nones
    df = nan_count_features(df)
    df["cat_nan_count"] = df[cat_cols+ind_cols].isna().sum(axis=1).astype(float)
    df["bin_nan_count"] = df[bin_cols].isna().sum(axis=1).astype(float)
    df["num_nan_count"] = df[num_cols].isna().sum(axis=1).astype(float)
    df["cat_none_count"] = (df[cat_cols+ind_cols] == 'None').sum(axis=1).astype(float)
    df["bin_none_count"] = (df[bin_cols] == 'None').sum(axis=1).astype(float)
    df["num_none_count"] = (df[num_cols] == 'None').sum(axis=1).astype(float)
    # train_df = nan_count_features(train_df)
    # test_df = nan_count_features(test_df)

    # Replace some typos in macv and district/province
    # df.macv = df.macv.str.lower()
    df = handle_category_typo(df)
    # train_df = handle_category_typo(train_df)
    # test_df = handle_category_typo(test_df)

    # Fix age
    df["age_source"] = np.where(
        df["age_source2"] == df["age_source1"],
        df["age_source1"],
        np.nan,
    )
    num_cols.append('age_source')

    # Fix some fields
    df.loc[df["field_3"] == -1, "field_3"] = np.nan
    df.loc[
        df["field_9"].isin(["na", "79", "75", "80", "86"]), "field_9"
    ] = np.nan
    df["field_7"] = df["field_7"].fillna("[]").apply(lambda x: eval(x))

    # Add features for field_7
    df["field_7_elem_count"] = df["field_7"].apply(lambda x: len(x))
    f7_uniques = pd.DataFrame(df.field_7.tolist()).stack().unique()
    for c in f7_uniques:
        df[f"field_7_has_{c}"] = df.field_7.apply(lambda x: c in x)
        df[f"field_7_count_{c}"] = df.field_7.apply(lambda x: x.count(c))
    df["field_7"] = [",".join(map(str, l)) for l in df["field_7"]]

    # Handling binary cols
    new_bin_cols = []
    for c in bin_cols:
        df[f"{c}_is_null"] = df[c].isnull()
        df.loc[df[c] == 'None', c] = -1
        df[c] = df[c].astype("bool")
        df[c] = df[c].fillna(-999)
            
    # Handling cat cols
    for c in cat_cols:
        df[f"{c}_is_null"] = df[c].isnull()
        if df[c].isna().mean() > 0:  # Fill with unedfined
            df[c] = df[c].fillna("undefined")

    df['province_district'] = df['province'].astype('str') + df['district'].astype('str')
    cat_cols.append('province_district')

    # Handling ind cols
    df["field_41"].replace(roman_conversion, inplace=True)
    for c in ind_cols:
        df[f"{c}_is_null"] = df[c].isnull()
        df.loc[df[c] == "None", c] = -999
        if df[c].isna().mean() < 0.1:  # Fill with mode
            df[c] = df[c].fillna(df[c].mode()[0])
        else:  # replace with -1
            df[c] = df[c].fillna(-1)
        df[c] = df[c].astype(int)
        
            
    # New features Category Count feature
    # df, cat_count_cols = cat_count_features(df, cat_cols + ind_cols, normalise=False)

    # check if this value is unique in a column
    # u_cols = []
    # for c in num_cols + bin_cols + ind_cols:
    #     unique_v = df[c].value_counts()
    #     unique_v = unique_v.index[unique_v == 1]
    #     df[f"{c}_u"] = df[c].isin(unique_v)
    #     u_cols.append(f"{c}_u")
        
    # # Check if there is any 'unique' values in each row
    # df['has_unique'] = df[u_cols].any(axis=1)

    # # Sum of 'unique' values in each row
    # df['sum_unique'] = df[u_cols].sum(axis=1)

    # Ranking cols
    # df, rank_cols = rank_features(
    #     df, ["field_22", "field_51", "field_3"]
    # )
    # Fractional cols
    df, fractional_cols = fractional_features(
        df,
        ["field_51"],
    )

    group_cols = ["district", "field_7", "province"]
    agg_cols = [
        "field_3",
        "age_source2",
        "field_22",
        # "field_50",
        # "field_51",
        # "field_52",
        # "field_53",
        # "field_54",
        # "field_55",
    ]

    # Agg features
    # df, agg_ft = aggregate_features(df, group_cols, agg_cols)

    # Interaction features
    feature_set = ["age_source1", "age_source2", "field_3", "field_22"]
    # df, inter_cols = interact_features(df, feature_set)

    # Label encoding
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(df[c].astype(str))
        df[c] = le.transform(df[c].astype(str))
    # Categories with less than or eq 10 unique values
    # cat_cols_less10 = [c for c in cat_cols if df[c].nunique() < 10]

    # One-hot encoded cols
    # df = pd.get_dummies(df, columns=cat_cols_less10, drop_first=False)

    # New feature: Sum of binary
    df["bin_sum"] = df[new_bin_cols].sum(axis=1).astype(float)
    # features.append("bin_sum")
    # num_cols.append("bin_sum")

    # New features: Differences measure for binary feature
    # Reference row for binary features
    bin_ref = df[new_bin_cols].copy()
    bin_ref_med = bin_ref.median(axis=0).to_frame().T
    bin_ref_med = pd.DataFrame(
        bin_ref_med.values.repeat(df.shape[0], axis=0),
        columns=bin_ref_med.columns,
    )
    # Calculate the difference
    df["bin_diff"] = (
        df[new_bin_cols].subtract(bin_ref_med).abs().sum(axis=1).astype(float)
    )
    # features.append("bin_diff")
    # num_cols.append("bin_diff")

    # Mean of 2 age sources
    df["mean_age"] = df[["age_source1", "age_source2"]].mean(axis=1)

    # Handling numeric cols
    for c in num_cols:
        df.loc[df[c] == "None", c] = np.nan
        if df[c].isna().mean() < 0.1:  # Fill with median
            df[c] = df[c].fillna(df[c].median())
        # else:  # replace with -1
            # df[c] = df[c].fillna(-1)

    # df = df.drop(
    #     columns=[c for c in cat_cols if c not in cat_cols_less10]
    # )
    # Prepare input
    logging.info("Preparing input...")
    features = df.drop(columns=['id', 'label']).columns
    y = df.loc[df['id'] < 30000, 'label']
    X = df.loc[df['id'] < 30000, features]
    X_test = df.loc[df['id'] >= 30000, features]
    X_test_id = df.loc[df['id'] >= 30000, 'id'].astype('int')
    print("Training data is")
    print(X.shape)
    print(y.shape)
    # print(X.head())
    print("Test data is")
    print(X_test.shape)
    # print(X_test.head())
    # sys.exit()
    logging.info("Modelling...")

    # Modelling
    # cv 0.12980138812313147
    # xgb_params = {
    #     'colsample_bytree': 0.3166676771353881, 
    #     'min_child_weight': 20, 
    #     'gamma': 1, 
    #     'subsample': 0.9209385587212306, 
    #     'max_depth': 8, 
    #     'n_estimators': 636, 
    #     'learning_rate': 0.011090841102396111
    # }
    # CV  0.13327110809193118
    xgb_params = {
        'n_jobs': -1,
        'colsample_bytree': 0.511182194059105, 
        'min_child_weight': 5, 
        'gamma': 0, 
        'subsample': 0.7937217003281334, 
        'max_depth': 5, 
        'n_estimators': 231, 
        'learning_rate': 0.044806747122933484
    }

    # cv 0.14070206805613594
    # xgb_params={'n_jobs': -1, 'colsample_bytree': 0.5375433726590078, 'min_child_weight': 5, 'gamma': 1, 'subsample': 0.7215565097969526, 'max_depth': 3, 'n_estimators': 1618, 'learning_rate': 0.03727028793820651, 'scale_pos_weight': 9.204985495237583}


    res_df = pd.DataFrame({'id': X_test_id, 'label': 0})
    # X, X_test = mean_encoding(X, y, X_test, ['field_3', 'field_22', 'field_9', 'field_19', 'macv', 'province_district'])
    # n_experiments = 5
    # for i in tqdm(range(n_experiments)):
    #     np.random.seed(seed=int(time.time()))
    #     clf = xgb.XGBClassifier(**xgb_params)
    #     clf.fit(
    #         X, 
    #         y, 
    #         verbose=False,
    #         # eval_set=[(Xtest, ytest)],
    #         # eval_metric=gini_xgb,
    #         # early_stopping_rounds=100,
    #     )
    #     res_df['label'] += clf.predict_proba(X_test)[:, 1]
    # res_df['label'] /= n_experiments
    
    # clf.fit(X, y)
    scores = []
    mean_ft = ['field_3', 'field_22', 'field_9', 'field_19', 'macv', 'province_district', 'field_1', 'field_6']
    label_cols = []
    n_seeds = 10
    for i, (train_index, test_index) in enumerate(folds):
        print(f"Fold {i+1}")
        label_cols.append(f'label_{i}')
        res_df[f'label_{i}'] = 0
        fold_scores = []
        Xtrain, Xtest = X.iloc[train_index].copy().reset_index(drop=True), X.iloc[test_index].copy().reset_index(drop=True)
        ytrain, ytest = y[train_index].copy().reset_index(drop=True), y[test_index].copy().reset_index(drop=True)
        for seed in tqdm(range(n_seeds)):
            # seed = rn.randint(1,1000)
            # print(f"seed {seed}")
            # print("Mean encoding...")

            Xtrain, Xtest = mean_encoding(Xtrain, ytrain, Xtest, mean_ft)
            np.random.seed(seed=int(time.time()))
            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(
                Xtrain, 
                ytrain, 
                verbose=False,
                eval_set=[(Xtest, ytest)],
                eval_metric=gini_xgb,
                early_stopping_rounds=100,
            )
            # print("Best N trees = ", clf.best_ntree_limit )
            # print("Best gini = ", clf.best_score )
            # clf.fit(Xtrain, ytrain)
            Xtrain, X_test = mean_encoding(Xtrain, ytrain, X_test, mean_ft)
            res_df[f'label_{i}'] += clf.predict_proba(X_test)[:, 1]
            
            ypred = clf.predict_proba(Xtest)[:, 1]
            score = gini(ytest, ypred)
            # print(f"Score: {score}")
            fold_scores.append(score)
            scores.append(score)
        res_df[f"label_{i}"] = res_df[f"label_{i}"].values / n_seeds
        print(f"Fold score: {np.mean(fold_scores)}")
    print(f"Mean score {np.mean(scores)}")

    # Averaging
    res_df['label'] = res_df[label_cols].mean(axis=1).round(6)
    print(res_df.head(1))

    # logging.info("Saving csv...")
    res_df[['id', 'label']].to_csv('submissions/sub_xgb10_1_cv.csv', index=False)


if __name__ == '__main__':
    main()

