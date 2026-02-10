from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

BASE_MODELS = {
    "lr": LogisticRegression(max_iter=500, random_state=42),
    "gb": GradientBoostingClassifier(n_estimators=100, random_state=42)
}


__all__ = [
    BASE_MODELS
]