import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


class GradientBoostingRandomSubspace(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        base_model,
        n_estimators=100,
        learning_rate=0.1,
        max_features=0.7,
        random_state=42,
        verbose=True
    ):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.init_val_ = np.log(p / (1 - p))

        F = np.full(n_samples, self.init_val_)

        self.models_ = []
        self.feature_sets_ = []

        iterator = range(self.n_estimators)
        if self.verbose:
            iterator = tqdm(iterator, desc="Boosting")

        for _ in iterator:

            probs = 1 / (1 + np.exp(-F))
            residuals = y - probs

            k = max(1, int(self.max_features * n_features))
            features = rng.choice(n_features, size=k, replace=False)

            model = clone(self.base_model)
            model.fit(X[:, features], residuals)

            update = model.predict(X[:, features])
            F += self.learning_rate * update

            self.models_.append(model)
            self.feature_sets_.append(features)

        return self

    def predict_proba(self, X):

        X = np.asarray(X)
        F = np.full(X.shape[0], self.init_val_)

        for model, features in zip(self.models_, self.feature_sets_):
            F += self.learning_rate * model.predict(X[:, features])

        probs = 1 / (1 + np.exp(-F))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def score_metrics(y_prob, y, verbose=True):
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        "confusion_matrix": confusion_matrix(y, y_pred)
    }

    if verbose:
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])

    return metrics
