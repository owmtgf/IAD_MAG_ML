import pandas as pd
from preprocessing_scripts import MISSING_TOKENS


def get_mapping(df: pd.DataFrame) -> dict:

    binary_mapping = {}

    for col in df.columns:
        if df[col].dtype != "object":
            continue

        unique_vals = (
            df[col]
            .dropna()
            .loc[~df[col].isin(MISSING_TOKENS)]
            .unique()
        )

        sorted_vals = sorted(unique_vals)

        binary_mapping[col] = {
            sorted_vals[i]: i for i in range(len(sorted_vals))
        }

    return binary_mapping


def flip_mapping(mapping: dict) -> dict:

    flipped = {}

    for feature, feature_map in mapping.items():
        assert isinstance(feature_map, dict), (
            f"Mapping for feature '{feature}' must be a dict"
        )

        # Ensure values are unique (invertible)
        values = list(feature_map.values())
        assert len(values) == len(set(values)), (
            f"Mapping for feature '{feature}' is not invertible"
        )

        flipped[feature] = {v: k for k, v in feature_map.items()}

    return flipped