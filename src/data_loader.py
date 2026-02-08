"""
Unified Data Loader Module for Hate Speech Detection.

This module implements a unified data loading interface for multiple hate speech
datasets, ensuring consistent preprocessing and column naming across different
data sources. Supports both gutefrage.net and HOCON34k datasets.

Unified Output Format:
    - text: The text content to classify
    - label: Binary label (1 = Hate Speech, 0 = Not Hate Speech)
    - source_info: Metadata about the sample origin

Author: Bachelor Thesis Project
Date: 2025
"""

import logging
from typing import Literal, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_prepare_data(
    filepath: str,
    dataset_type: Literal['gutefrage', 'hocon'] = 'gutefrage',
    sample_size: Optional[int] = 50
) -> pd.DataFrame:
    """
    Load and prepare hate speech dataset from multiple sources.

    This function implements a unified data loading interface that normalizes
    different dataset formats into a consistent schema for downstream processing.
    Supports balanced sampling to ensure equal representation of both classes.

    Args:
        filepath: Path to the CSV dataset file
        dataset_type: Dataset format - either 'gutefrage' or 'hocon'
        sample_size: Number of samples to return (balanced 50/50 split).
                    If None, returns full dataset.

    Returns:
        DataFrame with standardized columns:
            - text (str): The text content to classify
            - label (int): Binary label (1=HS, 0=NOT-HS)
            - source_info (str): Metadata about sample origin

    Raises:
        ValueError: If dataset_type is not recognized
        FileNotFoundError: If filepath does not exist
        KeyError: If required columns are missing from dataset

    Example:
        >>> df = load_and_prepare_data('data/raw/gutefrage.csv', 'gutefrage', 100)
        >>> print(df.columns)
        Index(['text', 'label', 'source_info'], dtype='object')
    """
    logging.info(f"Loading {dataset_type} dataset from: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logging.warning(f"Failed with default separator, trying ';': {e}")
        df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')

    if dataset_type == 'gutefrage':
        return _prepare_gutefrage(df, sample_size)
    elif dataset_type == 'hocon':
        return _prepare_hocon(df, sample_size)
    else:
        raise ValueError(f"Unknown dataset_type: '{dataset_type}'. Must be 'gutefrage' or 'hocon'")


def _prepare_gutefrage(df: pd.DataFrame, sample_size: Optional[int]) -> pd.DataFrame:
    """
    Prepare gutefrage.net dataset with deletion_reason mapping.

    The gutefrage dataset contains moderation deletion reasons. This function
    maps specific reasons to hate speech labels based on domain knowledge:
    - Hate Speech (1): Personal attacks, hostility toward groups, violence incitement
    - Not Hate Speech (0): Spam, trolling, off-topic content

    Args:
        df: Raw gutefrage DataFrame
        sample_size: Number of balanced samples to return

    Returns:
        Standardized DataFrame with columns: text, label, source_info
    """
    logging.info("Preparing gutefrage.net dataset...")

    # Domain-specific hate speech classification based on deletion reasons
    hs_reasons = [
        "Nutzervorführung / Persönlicher Angriff",
        "Feindseligkeit gegenüber Dritten",
        "Aufruf zu Gewalt / Straftaten"
    ]

    def map_label(reason: str) -> int:
        """Map deletion reason to binary hate speech label."""
        return 1 if reason in hs_reasons else 0

    # Data cleaning and preprocessing
    df = df.dropna(subset=['body', 'deletion_reason'])
    df['label'] = df['deletion_reason'].apply(map_label)

    # Rename and select columns for unified format
    data = df[['body', 'label', 'deletion_reason']].rename(columns={'body': 'text'})
    data['source_info'] = data['deletion_reason']

    # Balanced sampling with reproducibility
    if sample_size:
        data = _balanced_sample(data, sample_size, random_state=42)

    logging.info(f"Gutefrage dataset loaded: {len(data)} samples")
    logging.info(f"  Hate Speech samples: {sum(data['label'] == 1)}")
    logging.info(f"  Not Hate Speech samples: {sum(data['label'] == 0)}")

    return data[['text', 'label', 'source_info']].reset_index(drop=True)


def _prepare_hocon(df: pd.DataFrame, sample_size: Optional[int]) -> pd.DataFrame:
    """
    Prepare HOCON34k dataset with label_hs mapping.

    The HOCON dataset is pre-labeled by expert annotators with binary
    hate speech labels. This function normalizes it to the unified format.

    Args:
        df: Raw HOCON DataFrame
        sample_size: Number of balanced samples to return

    Returns:
        Standardized DataFrame with columns: text, label, source_info
    """
    logging.info("Preparing HOCON34k dataset...")

    # Validate required columns
    required_cols = ['text', 'label_hs']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"HOCON dataset missing required columns: {missing_cols}")

    # Data cleaning and preprocessing
    df = df.dropna(subset=['text', 'label_hs'])
    df['label'] = df['label_hs'].astype(int)

    # Create source_info from label category
    df['source_info'] = df.apply(
        lambda row: f"HOCON_HS" if row['label'] == 1 else "HOCON_NOT-HS",
        axis=1
    )

    data = df[['text', 'label', 'source_info']]

    # Balanced sampling with reproducibility
    if sample_size:
        data = _balanced_sample(data, sample_size, random_state=42)

    logging.info(f"HOCON dataset loaded: {len(data)} samples")
    logging.info(f"  Hate Speech samples: {sum(data['label'] == 1)}")
    logging.info(f"  Not Hate Speech samples: {sum(data['label'] == 0)}")

    return data.reset_index(drop=True)


def _balanced_sample(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create balanced sample with equal representation from both classes.

    Current implementation uses random stratified sampling. This ensures class
    balance and reproducibility via random_state seeding.

    Args:
        df: DataFrame with 'label' column
        sample_size: Total number of samples (will be split 50/50)
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame with sample_size samples
    """
    # Calculate balanced sample sizes
    samples_per_class = sample_size // 2

    # Sample from each class
    pos_samples = df[df['label'] == 1]
    neg_samples = df[df['label'] == 0]

    # Handle cases where class has fewer samples than requested
    n_pos = min(samples_per_class, len(pos_samples))
    n_neg = min(samples_per_class, len(neg_samples))

    if n_pos < samples_per_class or n_neg < samples_per_class:
        logging.warning(
            f"Insufficient samples for balanced split. "
            f"Requested: {samples_per_class}/class, "
            f"Available: {n_pos} HS, {n_neg} NOT-HS"
        )

    # Random balanced sampling (current implementation)
    pos_sample = pos_samples.sample(n=n_pos, random_state=random_state)
    neg_sample = neg_samples.sample(n=n_neg, random_state=random_state)

    # Combine and shuffle
    balanced_data = pd.concat([pos_sample, neg_sample])
    balanced_data = balanced_data.sample(frac=1, random_state=random_state)

    return balanced_data


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified validation/test split to prevent data leakage.

    CRITICAL FOR SCIENTIFIC VALIDITY:
    This function ensures that the optimization loop (Phase 2) never sees the test
    set, preventing the prompt from overfitting to evaluation data. The split is
    stratified to maintain class balance across both sets.

    Data Usage Protocol:
        - val_df: Used for optimization loop (error analysis, metric calculation)
        - test_df: HELD OUT completely. Used ONCE at the end for final reporting.

    Note on train_df removal:
        The original three-way split included a train_df for future few-shot sampling.
        Since the current implementation uses zero-shot + error-driven refinement,
        train_df was removed to simplify the codebase and prevent confusion.

    Args:
        df: Full dataset DataFrame with 'label' column
        val_size: Fraction of data for validation set (default: 0.2 = 20%)
        test_size: Fraction of data for test set (default: 0.2 = 20%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (val_df, test_df) with stratified class distribution

    Example:
        >>> df = load_and_prepare_data('data.csv', sample_size=100)
        >>> val, test = split_data(df, val_size=0.5, test_size=0.5)
        >>> # val: 50 samples, test: 50 samples

    Raises:
        ValueError: If val_size + test_size > 1.0
    """
    if val_size + test_size > 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must be <= 1.0"
        )

    logging.info("=" * 70)
    logging.info("STRATIFIED DATA SPLIT (Preventing Data Leakage)")
    logging.info("=" * 70)
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Split ratio: Val={val_size:.0%} / Test={test_size:.0%}")

    # Clean two-way split: validation and test
    # When val_size + test_size = 1.0, this uses 100% of data with ZERO waste
    # train_test_split returns (1-test_size, test_size) by default
    val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )

    # Log split statistics
    logging.info("\nVALIDATION SET (used for optimization loop):")
    logging.info(f"  Total: {len(val_df)} samples")
    logging.info(f"  HS: {sum(val_df['label'] == 1)} | NOT-HS: {sum(val_df['label'] == 0)}")

    logging.info("\nTEST SET (held out for final evaluation):")
    logging.info(f"  Total: {len(test_df)} samples")
    logging.info(f"  HS: {sum(test_df['label'] == 1)} | NOT-HS: {sum(test_df['label'] == 0)}")

    # Data integrity verification (CRITICAL: ensure zero data waste)
    total_split = len(val_df) + len(test_df)
    logging.info(f"\nData Integrity Check:")
    logging.info(f"  Original: {len(df)} samples")
    logging.info(f"  Split total: {total_split} samples (Val: {len(val_df)} + Test: {len(test_df)})")
    logging.info(f"  Data waste: {len(df) - total_split} samples")

    if total_split != len(df):
        logging.warning(f"  ⚠️  DATA WASTE DETECTED: {len(df) - total_split} samples discarded!")
    else:
        logging.info(f"  ✅ Zero data waste - all {len(df)} samples utilized")

    logging.info("=" * 70)

    return val_df.reset_index(drop=True), test_df.reset_index(drop=True)
