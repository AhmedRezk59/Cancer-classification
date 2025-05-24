import logging
import pandas as pd
import numpy as np

logger: logging.Logger = logging.getLogger(__name__)
pd.set_option('display.max_columns', 20)
pd.set_option("display.max_rows", 101)

def analyze_dataframe(dataframe: pd.DataFrame)->None:
    """
    Analyze the dataframe and print the results.
    :param dataframe: The dataframe to analyze.
    :rtype: None
    """
    logger.info("Running Exploratory Data Analysis")
    print(dataframe.head())
    print(dataframe.shape)
    print(dataframe.dtypes)
    print(dataframe.info(memory_usage="deep"))
    print(dataframe.memory_usage(deep=True))
    print(dataframe.describe(include="all" , datetime_is_numeric=True))
    non_numeric_df = dataframe.select_dtypes(exclude=[np.number])
    for column in non_numeric_df.columns:
        print(non_numeric_df[column].value_counts())
        print(non_numeric_df[column].unique())
        print(non_numeric_df[column].value_counts(normalize=True))

def find_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Find missing values in the dataframe and print the results.
    :param dataframe: The dataframe to analyze.
    :rtype: pd.DataFrame
    """
    logger.info("Finding missing values")
    missing_values = dataframe.isnull().sum()
    print(missing_values)
    if missing_values.any():
        logger.warning("Missing values found in the following columns:")
        for column, count in missing_values.items():
            if count > 0:
                logger.warning(f"{column}: {count} missing values")
    return dataframe.dropna()  # Example of handling missing values