from __future__ import annotations

"""
Utilities for loading and preprocessing the RAID dataset.

RAID CSV schema (key columns):
    id              - unique generation ID
    generation      - the text (human or AI-generated)
    model           - generating model name (or "human")
    domain          - text domain (news, reddit, arxiv, book, wiki, poetry, recipes, imdb)
    attack          - adversarial attack applied (or "none")
    decoding        - decoding strategy used
    label           - 0 = human, 1 = AI-generated
    split           - train / test
"""

from pathlib import Path

import pandas as pd

DOMAIN_VALUES = [
    "news", "reddit", "arxiv", "book", "wikipedia",
    "poetry", "recipes", "imdb",
]

MODEL_VALUES = [
    "human", "gpt2", "chatgpt", "gpt-4", "llama-chat",
    "mistral", "mistral-chat", "mpt", "mpt-chat", "cohere",
]


def load_split(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a single RAID CSV file and return a cleaned DataFrame."""
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    # Derive label: 0 = human, 1 = AI (train has model col; test labels are withheld)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    elif "model" in df.columns:
        df["label"] = (df["model"].str.lower() != "human").astype(int)
    return df


def load_raid(
    data_dir: str | Path = "data/raw",
    train_file: str = "train.csv",
    test_file: str = "test.csv",
    train_nrows: int | None = None,
    test_nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the RAID train and test splits.

    Parameters
    ----------
    data_dir    : path to the directory containing the CSV files.
    train_file  : filename of the training split.
    test_file   : filename of the test split.
    train_nrows : if set, load only the first N rows of train (useful for EDA).
    test_nrows  : if set, load only the first N rows of test.

    Returns
    -------
    (train_df, test_df) as a tuple of DataFrames.
    """
    data_dir = Path(data_dir)
    train_df = load_split(data_dir / train_file, nrows=train_nrows)
    test_df = load_split(data_dir / test_file, nrows=test_nrows)
    return train_df, test_df


def filter_by_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Return rows matching a specific domain."""
    return df[df["domain"].str.lower() == domain.lower()].copy()


def filter_by_model(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return rows for a specific generating model (or 'human')."""
    return df[df["model"].str.lower() == model.lower()].copy()


def filter_no_attack(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where no adversarial attack was applied."""
    return df[df["attack"].str.lower() == "none"].copy()
