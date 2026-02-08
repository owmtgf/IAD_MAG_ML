import numpy as np
import pandas as pd
from preprocessing_scripts import MISSING_TOKENS


def normalize_missing(df, features):
    df = df.copy()
    df[features] = df[features].replace(MISSING_TOKENS, np.nan)
    return df


def fit_age_imputer(train_df: pd.DataFrame) -> dict:

    return {
        "idh1": (
            train_df
            .groupby("IDH1")["Age_at_diagnosis"]
            .median()
            .to_dict()
        ),
        "global": train_df["Age_at_diagnosis"].median()
    }


def apply_age_imputer(
    df: pd.DataFrame,
    age_stats: dict
) -> pd.DataFrame:

    df = df.copy()

    missing_mask = df["Age_at_diagnosis"].isna()

    for idx in df[missing_mask].index:
        idh1 = df.at[idx, "IDH1"]

        if idh1 in age_stats["idh1"]:
            df.at[idx, "Age_at_diagnosis"] = age_stats["idh1"][idh1]
        else:
            df.at[idx, "Age_at_diagnosis"] = age_stats["global"]

    return df