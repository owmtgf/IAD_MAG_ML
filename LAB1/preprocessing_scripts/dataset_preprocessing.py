import pandas as pd
import numpy as np
import re


def print_unique_values(df: pd.DataFrame, max_values: int=20):

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)
        
        print(f"\nColumn: {col}")
        print(f"Number of unique values: {n_unique}")
        
        if n_unique <= max_values:
            print("Unique values:", unique_vals)
        else:
            print(f"First {max_values} unique values:", unique_vals[:max_values])
            print("... (truncated)")


def drop_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    
    missing_cols = [col for col in feature_names if col not in df.columns]
    assert len(missing_cols) == 0, (
        f"The following columns do not exist in DataFrame: {missing_cols}"
    )
    
    return df.copy().drop(columns=feature_names)


def map_features(df: pd.DataFrame, feature_mapping: dict) -> pd.DataFrame: 

    df_out = df.copy()
    
    for feature, mapping in feature_mapping.items():
        assert feature in df_out.columns, (
            f"Feature '{feature}' not found in DataFrame"
        )
        assert isinstance(mapping, dict), (
            f"Mapping for feature '{feature}' must be a dict"
        )
        
        existing_values = set(df_out[feature].dropna().unique())
        mapping_keys = set(mapping.keys())
        
        invalid_keys = mapping_keys - existing_values
        assert len(invalid_keys) == 0, (
            f"Mapping for feature '{feature}' contains labels "
            f"not present in data: {invalid_keys}"
        )
        
        df_out[feature] = df_out[feature].map(
            lambda x: mapping.get(x, x)
        )
    
    return df_out


def fill_not_reported_field(
    df: pd.DataFrame,
    feature: str = "Race",
    mode: str = "mode",
    random_state: int = 42
) -> pd.DataFrame:
    
    assert feature in df.columns, f"Feature '{feature}' not found in DataFrame"
    assert mode in {"mode", "distribution"}, "mode must be 'mode' or 'distribution'"

    df_out = df.copy()

    # Define missing-like values
    missing_tokens = {"--", "not reported"}

    # Mask for missing values
    missing_mask = df_out[feature].isin(missing_tokens)
    n_missing = missing_mask.sum()

    if n_missing == 0:
        return df_out

    # Valid (observed) values
    valid_values = df_out.loc[~missing_mask, feature]

    assert valid_values.nunique() > 0, (
        f"No valid values available to impute '{feature}'"
    )

    if mode == "mode":
        # Most frequent value
        fill_value = valid_values.mode().iloc[0]
        df_out.loc[missing_mask, feature] = fill_value

    elif mode == "distribution":
        value_probs = valid_values.value_counts(normalize=True)

        rng = np.random.default_rng(seed=random_state)
        sampled_values = rng.choice(
            value_probs.index,
            size=n_missing,
            p=value_probs.values
        )

        df_out.loc[missing_mask, feature] = sampled_values

    return df_out


def split_feature_components(df: pd.DataFrame, feature: str = "Primary_Diagnosis") -> pd.DataFrame:

    assert feature in df.columns, f"Feature '{feature}' not found in DataFrame"

    df_out = df.copy()

    # Define missing-like tokens
    missing_tokens = {"--"}

    # Extract all unique values excluding missing
    unique_values = [v for v in df_out[feature].unique() if v not in missing_tokens]

    # Collect all unique terms
    terms_set = set()
    for val in unique_values:
        parts = [p.strip() for p in val.split(",")]
        terms_set.update(parts)
    
    terms_list = [t.replace(" ", "_") for t in terms_set]

    # Create boolean columns for each term
    for term in terms_list:
        original_term = term.replace("_", " ").lower()  # map back for matching
        df_out[f"{feature}_{term}"] = df_out[feature].apply(
            lambda x: 0 if x in missing_tokens else int(original_term in x.lower())
        )

    # Drop original column
    df_out = df_out.drop(columns=[feature])

    return df_out


def convert_lifetime(df: pd.DataFrame, feature: str = "Age_at_diagnosis") -> pd.DataFrame:

    assert feature in df.columns, f"Feature '{feature}' not found in DataFrame"
    
    df_out = df.copy()
    
    def parse_age(value):
        if pd.isna(value):
            return np.nan
        # Use regex to extract numbers
        match = re.match(r'(?:(\d+)\s*years)?\s*(?:(\d+)\s*days)?', str(value))
        if match:
            years = int(match.group(1)) if match.group(1) else 0
            days = int(match.group(2)) if match.group(2) else 0
            return years * 365 + days
        else:
            raise ValueError(f"Got an unexpexted format \"{str(value)}\"")
    
    df_out[feature] = df_out[feature].apply(parse_age)
    
    return df_out