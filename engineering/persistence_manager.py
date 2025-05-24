from enum import Enum
from typing import Union,Optional
import pandas as pd
from core.config import ENCODING 
import os

class DataType(Enum):
    """
    Enum for data types.
    """
    RAW:str = "data/raw"
    PROCESSED:str = "data/processed"
    FIGURES:str = "reports/figures/"

class PersistenceManager:
    """
    Class to manage the persistence of data in the project.
    """
    @staticmethod
    def save_to_csv(
        data: Union[list[dict] , pd.DataFrame],
        data_type:DataType = DataType.PROCESSED,
        filename:str = "data"        
    ) -> bool:
        """
        Saves data to a CSV file.
        Args:
            data (Union[list[dict], pd.DataFrame]): Data to be saved.
            data_type (DataType, optional): Type of data. Defaults to DataType.PROCESSED.
            filename (str, optional): Name of the file. Defaults to "data".
        rtype: bool
        """
        dataframe:pd.DataFrame
        if isinstance(data,pd.DataFrame):
            dataframe = data
        else:
            if not data:
                return False
            dataframe = pd.DataFrame(data)
        dataframe.to_csv(
            f"{str(data_type)}{filename}.csv",
            index=False ,
            encoding = ENCODING
        )
        return True
    
    @staticmethod
    def load_from_csv(
        filename:str,
        data_type:DataType,
        chunk_size :int,
        dtypes:Optional[dict],
        converter: Optional[dict]
    ) -> pd.DataFrame:
        """
        Loads data from a CSV file.
        Args:
            filename (str): Name of the file.
            data_type (DataType): Type of data.
            chunk_size (int): Size of the chunks to be loaded.
            dtypes (Optional[dict], optional): Data types of the columns. Defaults to None.
            converter (Optional[dict], optional): Converter for the columns. Defaults to None.
        rtype: pd.DataFrame
        """
        filepath :str= f"{data_type.value}/{filename}"
        text_file_reader : TextFileReader = pd.read_csv(
            filepath,
            header=0,
            chunksize = chunk_size,
            dtype = dtypes,
            converters = converter,
            encoding = ENCODING
        )
        dataframe:pd.DataFrame = pd.concat(
            text_file_reader,
            ignore_index=True
        )
        if dtypes:
            for key,value in dtypes.items():
                if value in (int ,float):
                    dataframe[key] = pd.to_numeric(
                        dataframe[key] , errors="coerce"
                    )
                    dataframe[key] = dataframe[key].astype(value)
                else:
                    dataframe[key] = dataframe[key].astype(value)
        return dataframe

    @staticmethod
    def save_to_pickle(
        dataframe:pd.DataFrame,
        filename:str = 'optimized_df.pkl'
    ) ->None:
        """
        Saves a pandas DataFrame to a pickle file.
        Args:
            dataframe (pd.DataFrame): DataFrame to be saved.
            filename (str, optional): Name of the file. Defaults to 'optimized_df.pkl'.
        rtype: None
        """
        os.path.dirname(f'data/processed/{filename}')
        if not os.path.exists(f'data/processed/'):
            os.makedirs(f'data/processed/')
        dataframe.to_pickle(f'data/processed/{filename}')
        
    @staticmethod
    def load_from_pickle(filename: str = 'optimized_df.pkl') -> pd.DataFrame:
        """
        Load dataframe from Pickle file
        :param filename: name of the file to search and load
        :type filename: str
        :return: dataframe read from pickle
        :rtype: pd.DataFrame
        """
        dataframe: pd.DataFrame = pd.read_pickle(f'data/processed/{filename}')
        return dataframe    