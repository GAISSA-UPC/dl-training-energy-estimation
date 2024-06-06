import pandas as pd


def encode_dataset(dataset: pd.DataFrame):
    """
    Encode categorical features in a dataset using the `category` dtype.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to encode.

    Returns
    -------
    pd.DataFrame
        The encoded dataset.
    dict
        A dictionary mapping each categorical feature to its categories.
    """
    categories_mapping = {}
    encoded_df = dataset.copy()
    for categorical_feature in encoded_df.select_dtypes(include="object").columns:
        categories_mapping[categorical_feature] = encoded_df[categorical_feature].astype("category").cat.categories
        encoded_df[categorical_feature] = encoded_df[categorical_feature].astype("category").cat.codes
    return encoded_df, categories_mapping
