import pandas as pd
from sklearn.base import clone

from meta_classifier import BASE_MODELS


def train_pure_meta_model(meta_model, meta_X: pd.DataFrame, y: pd.Series):

    meta_model.fit(meta_X, y)
    return meta_model

def infer_pure_meta_model(pure_meta_model, X_train, y_train, X_test):
    meta_test = pd.DataFrame(index=X_test.index)
    for name, model in BASE_MODELS.items():
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        meta_test[name] = model_clone.predict_proba(X_test)[:, 1]

    y_test_pure = pure_meta_model.predict_proba(meta_test)[:, 1]
    return y_test_pure


def train_meta_augmented_model(augmented_model, real_X: pd.DataFrame, meta_X: pd.DataFrame, y: pd.Series):

    X_train_aug = pd.concat(
        [real_X.reset_index(drop=True),
        meta_X.reset_index(drop=True)],
        axis=1
    )

    augmented_model.fit(X_train_aug, y)
    return augmented_model


def infer_meta_augmented_model(meta_augmented_model, X_train, y_train, X_test):
    meta_test = pd.DataFrame(index=X_test.index)
    for name, model in BASE_MODELS.items():
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        meta_test[name] = model_clone.predict_proba(X_test)[:, 1]

    X_test_aug = pd.concat(
        [X_test.reset_index(drop=True),
        meta_test.reset_index(drop=True)],
        axis=1
    )

    y_test_aug = meta_augmented_model.predict_proba(X_test_aug)[:, 1]
    return y_test_aug
