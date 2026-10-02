from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "Iris.csv"

FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

TARGET_COLUMN = "Species"


def load_dataset():
    """Load and clean the Iris flower dataset."""

    df = pd.read_csv(DATA_PATH)

    # Remove duplicate records and incomplete rows
    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Convert flower measurements into numeric values
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS)

    # Remove unnecessary identifier column
    df = df.drop(columns=["Id"], errors="ignore")

    # Clean species names for display
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.replace("Iris-", "", regex=False)
        .str.strip()
        .str.title()
    )

    return df.reset_index(drop=True)


def prepare_data(test_size=0.2, random_state=42):
    """Prepare features, labels, and training datasets."""

    df = load_dataset()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_labels
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "dataframe": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": FEATURE_COLUMNS,
        "class_names": encoder.classes_.tolist()
    }