import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from meta_classifier import BASE_MODELS


def create_meta_dataset(X: pd.DataFrame, y: pd.Series, n_splits=5, base_models=BASE_MODELS):

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_preds = {}
    
    for name in base_models.keys():
        meta_preds[f"{name}_prob"] = np.zeros(len(X))
        meta_preds[f"{name}_margin"] = np.zeros(len(X))
    
    for name, model in base_models.items():
        print(f"Generating OOF predictions for {name}...")
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            model_fold = clone(model)
            model_fold.fit(X_train, y_train)

            proba = model_fold.predict_proba(X_val)[:, 1]
            meta_preds[f"{name}_prob"][val_idx] = proba
            
            # margin / logit / decision function
            if hasattr(model_fold, "decision_function"):
                margin = model_fold.decision_function(X_val)
            else:
                eps = 1e-6
                margin = np.log(np.clip(proba, eps, 1 - eps) / np.clip(1 - proba, eps, 1 - eps))
            
            meta_preds[f"{name}_margin"][val_idx] = margin

    meta_X = pd.DataFrame(meta_preds, index=X.index)
    meta_y = y.copy()
    return meta_X, meta_y


def remove_highly_correlated(meta_X: pd.DataFrame, corr_thresh: float = 0.95) -> pd.DataFrame:

    meta_X_clean = meta_X.copy()
    corr_matrix = meta_X_clean.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    
    for col in upper.columns:
        if any(upper[col][~upper.index.isin(to_drop)] > corr_thresh):
            to_drop.append(col)
    
    if to_drop:
        print(f"Dropping highly correlated meta-features: {to_drop}")
    
    cleaned_meta_X = meta_X_clean.drop(columns=to_drop)
    return cleaned_meta_X