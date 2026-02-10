import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from meta_classifier import BASE_MODELS


def train_pure_meta_model(meta_model, meta_X: pd.DataFrame, y: pd.Series):

    meta_model.fit(meta_X, y)
    return meta_model


def infer_pure_meta_model(pure_meta_model, X_train, y_train, X_test):

    meta_test = pd.DataFrame(index=X_test.index)
    
    for name, model in BASE_MODELS.items():
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        proba = model_clone.predict_proba(X_test)[:, 1]
        meta_test[f"{name}_prob"] = proba
        
        if hasattr(model_clone, "decision_function"):
            margin = model_clone.decision_function(X_test)
        else:
            eps = 1e-6
            margin = np.log(np.clip(proba, eps, 1 - eps) / np.clip(1 - proba, eps, 1 - eps))
        
        meta_test[f"{name}_margin"] = margin

    y_test_pure = pure_meta_model.predict_proba(meta_test)[:, 1]
    return y_test_pure


def train_meta_augmented_model(augmented_model, real_X: pd.DataFrame, meta_X: pd.DataFrame, y: pd.Series, corr_thresh: float = 0.95):

    corr_matrix = meta_X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    meta_features_to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    meta_features = [col for col in meta_X.columns if col not in meta_features_to_drop]
    
    if meta_features_to_drop:
        print(f"Dropping highly correlated meta-features: {meta_features_to_drop}")
    
    meta_X_filtered = meta_X[meta_features].reset_index(drop=True)
    
    X_train_aug = pd.concat([real_X.reset_index(drop=True), meta_X_filtered], axis=1)
    
    scaler = StandardScaler()
    X_train_aug_scaled = pd.DataFrame(scaler.fit_transform(X_train_aug), columns=X_train_aug.columns)
    
    augmented_model.fit(X_train_aug_scaled, y)
    
    return augmented_model, scaler, meta_features


def infer_meta_augmented_model(meta_augmented_model, scaler, X_train, y_train, X_test, meta_features):

    meta_test_full = pd.DataFrame(index=X_test.index)
    
    for name, model in BASE_MODELS.items():
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        proba = model_clone.predict_proba(X_test)[:, 1]
        meta_test_full[f"{name}_prob"] = proba
        
        if hasattr(model_clone, "decision_function"):
            margin = model_clone.decision_function(X_test)
        else:
            eps = 1e-6
            margin = np.log(np.clip(proba, eps, 1 - eps) / np.clip(1 - proba, eps, 1 - eps))
        meta_test_full[f"{name}_margin"] = margin
    
    meta_test = meta_test_full[meta_features]
    
    X_test_aug = pd.concat([X_test.reset_index(drop=True), meta_test.reset_index(drop=True)], axis=1)
    
    X_test_aug = pd.DataFrame(scaler.transform(X_test_aug), columns=X_test_aug.columns)
    
    y_test_aug = meta_augmented_model.predict_proba(X_test_aug)[:, 1]
    return y_test_aug
