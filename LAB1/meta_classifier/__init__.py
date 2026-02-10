from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


BASE_MODELS = {
    "lr": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.1
    ),

    "gb": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.6,
        max_depth=3
    ),

    "rf": RandomForestClassifier(
        n_estimators=400,
        max_features=0.3,
        class_weight="balanced_subsample"
    ),

    "svm": SVC(
        kernel="rbf",
        C=10.0,
        probability=True,
        class_weight=None  # intentionally unbalanced
    ),

    "knn": KNeighborsClassifier(
        n_neighbors=15,
        weights="uniform"
    ),
}

__all__ = [
    BASE_MODELS
]