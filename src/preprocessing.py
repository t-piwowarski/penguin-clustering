import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Loads and cleans the penguin dataset.

    Cleaning steps:
    - Removes fully empty rows (e.g., culmen_length_mm is NaN)
    - Removes invalid flipper length values (< 0 or > 500)
    - Replaces '.' with NaN in the 'sex' column
    - Converts 'sex' values: 'MALE' -> 1, 'FEMALE' -> 0, others -> NaN

    Parameters:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = pd.read_csv(path)

    # Drop rows with missing culmen_length_mm
    df = df[~df['culmen_length_mm'].isna()]

    # Remove invalid flipper lengths
    df = df[~(df['flipper_length_mm'] < 0)]
    df = df[~(df['flipper_length_mm'] > 500)]

    # Replace '.' with NaN in 'sex' column
    df['sex'] = df['sex'].replace('.', np.nan)

    # Convert 'sex' to numeric: MALE=1, FEMALE=0, others=NaN
    df['sex'] = np.where(df['sex'] == 'MALE', 1,
                         np.where(df['sex'] == 'FEMALE', 0, np.nan))

    return df

def impute_missing_values(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Imputes missing values using K-Nearest Neighbors (KNN) imputer.

    Parameters:
        df (pd.DataFrame): DataFrame with missing values
        n_neighbors (int): Number of neighbors to use for imputation

    Returns:
        pd.DataFrame: Imputed DataFrame with rounded values
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(np.round(imputed_array), columns=df.columns)
    return df_imputed

def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the dataset using StandardScaler.
    After transformation, each feature has mean=0 and std=1.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Standardized DataFrame
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    standardized_df = pd.DataFrame(scaled_array, columns=df.columns)
    return standardized_df
