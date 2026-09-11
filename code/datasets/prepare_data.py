from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# Paths
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "penguins.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"


# =========================================================
# Settings
# =========================================================

TARGET = "species"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load raw dataset."""

    print("Loading dataset...")
    print(f"Path: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    # Make column names consistent
    df.columns = df.columns.str.strip().str.lower()

    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values, duplicates and unnecessary columns."""

    print("\n==============================")
    print("DATA CLEANING")
    print("==============================")

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Some versions of Palmer Penguins contain "." in the sex column
    if "sex" in df.columns:
        df["sex"] = df["sex"].replace(".", pd.NA)

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nDuplicates removed: {before - len(df)}")

    # ID does not describe a penguin.
    # Year is also not needed for our classifier.
    columns_to_drop = ["id", "year"]

    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns]
    )

    print(f"Dropped columns: {columns_to_drop}")

    # Check target
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # A row without a label cannot be used for supervised training
    df = df.dropna(subset=[TARGET])

    # Numerical features
    numeric_columns = df.drop(columns=[TARGET]).select_dtypes(
        include="number"
    ).columns

    # Fill missing numerical values with median
    for column in numeric_columns:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

    # Categorical features
    categorical_columns = df.drop(columns=[TARGET]).select_dtypes(
        exclude="number"
    ).columns

    # Fill missing categorical values with mode
    for column in categorical_columns:
        if df[column].isnull().any():
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove numerical outliers using the IQR method."""

    print("\n==============================")
    print("OUTLIER REMOVAL")
    print("==============================")

    numeric_columns = df.drop(columns=[TARGET]).select_dtypes(
        include="number"
    ).columns

    rows_before = len(df)

    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df = df[
            (df[column] >= lower_bound)
            & (df[column] <= upper_bound)
        ]

    rows_after = len(df)

    print(f"Rows before outlier removal: {rows_before}")
    print(f"Rows after outlier removal:  {rows_after}")
    print(f"Outliers removed:            {rows_before - rows_after}")

    return df


def split_data(df: pd.DataFrame):
    """Split cleaned dataset into train and test datasets."""

    print("\n==============================")
    print("TRAIN / TEST SPLIT")
    print("==============================")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")

    return train_df, test_df


def save_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Save processed datasets."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_DATA_PATH,
        index=False,
    )

    test_df.to_csv(
        TEST_DATA_PATH,
        index=False,
    )

    print("\n==============================")
    print("FILES SAVED")
    print("==============================")

    print(f"Train: {TRAIN_DATA_PATH}")
    print(f"Test:  {TEST_DATA_PATH}")


def main():
    df = load_data()

    df = clean_data(df)

    df = remove_outliers(df)

    train_df, test_df = split_data(df)

    save_data(train_df, test_df)

    print("\nStage 1 completed successfully.")


if __name__ == "__main__":
    main()