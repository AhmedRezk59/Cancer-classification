from engineering.persistence_manager import DataType, PersistenceManager
import pandas as pd
from typing import Optional
from core.config import CHUNK_SIZE
from numpy import uint8, uint16, float32
from engineering.transformation import convert_diagnostic_column

def extract_raw_data(
    filename:str = "RegistroCancer.csv",
    data_type:DataType = DataType.RAW,
    chunk_size : int = CHUNK_SIZE,
    d_types : Optional[dict] = None,
    converter: Optional[dict] = None,
    ) -> pd.DataFrame:
    """
    Extracts raw data from a CSV file and returns it as a pandas DataFrame.
    :param filename: The name of the CSV file to extract data from.
    :param data_type: The type of data to extract (e.g., RAW, PROCESSED).
    :param chunk_size: The number of rows to read at a time.
    :param d_types: A dictionary specifying the data types of the columns.
    :param converter: A dictionary specifying the conversion functions for the columns.
    :return: A pandas DataFrame containing the extracted data.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If the specified file does not exist.
    :raises ValueError: If the data type is not supported.
    :raises TypeError: If the chunk size is not an integer.
    :raises KeyError: If the specified column is not found in the DataFrame.
    :raises Exception: If an unexpected error occurs during data extraction.
    """
    
    if not d_types:
        d_types = {
            'id': uint8, 'radio': uint8, 'textura': uint8,
            'perímetro': uint8, 'área': uint16, 'suavidad': float32,
            'compacidad': float32, 'simetría': float32,
            'dimensión_fractal': float32}
    if not converter:
        converter:dict = {'resultado_diagnóstico': convert_diagnostic_column}
    dataframe: pd.DataFrame = PersistenceManager.load_from_csv(
        filename=filename,
        data_type=data_type,
        chunk_size=chunk_size,
        dtypes=d_types,
        converter=converter
    )
    return dataframe