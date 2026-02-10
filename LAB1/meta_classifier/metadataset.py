import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from meta_classifier import BASE_MODELS


def create_meta_dataset(X: pd.DataFrame, y: pd.Series, n_splits=5, base_models=BASE_MODELS):

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_preds = {name: np.zeros(len(X)) for name in base_models.keys()}

    for name, model in base_models.items():
        print(f"Generating predictions for {name}...")
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            model_fold = clone(model)

            model_fold.fit(X_train, y_train)
            proba = model_fold.predict_proba(X_val)[:, 1]
            meta_preds[name][val_idx] = proba

    meta_X = pd.DataFrame(meta_preds, index=X.index)
    meta_y = y.copy()
    return meta_X, meta_y
