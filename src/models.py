from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

def load_data():
    df = pd.read_csv('data/processed_data/processed_dataset.csv')
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

from sklearn.linear_model import LogisticRegression

def get_logistic_model():
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    return model

from sklearn.ensemble import RandomForestClassifier

def get_random_forest():
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )
    return model


from lightgbm import LGBMClassifier

def get_lightgbm_model():
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced"
    )
    return model

from xgboost import XGBClassifier

def get_xgboost_model():
    model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        tree_method="hist",
        random_state=42
    )
    return model

from catboost import CatBoostClassifier

def get_catboost_model():
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        verbose=0,
        random_seed=42
    )
    return model

def get_model(model_name: str):
    if model_name == "logistic":
        return get_logistic_model()

    elif model_name == "rf":
        return get_random_forest()

    elif model_name == "lgbm":
        return get_lightgbm_model()

    elif model_name == "xgb":
        return get_xgboost_model()

    elif model_name == "catboost":
        return get_catboost_model()

    else:
        raise ValueError("Unknown model name")
    
    


    
