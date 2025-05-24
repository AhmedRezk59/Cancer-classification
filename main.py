import logging
from core import logging_config
import pandas as pd
from engineering.data_extraction import extract_raw_data
from analysis import numerical_eda , visialize_data
from modelling.preprocessing import lof_observation
from engineering.persistence_manager import PersistenceManager
from models.models import iterate_models
import matplotlib

logging_config.setup_logging()
logger :logging.Logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main function to run the cancer classification project.
    rtype: None
    """
    matplotlib.use('Agg') 
    logger.info("Running main method")
    dataframe : pd.DataFrame = extract_raw_data()
    logger.info("Data extraction completed")
    dataframe = numerical_eda(dataframe)
    dataframe.drop("id" , axis=1 , inplace=True)
    logger.info("Dataframe shape after dropping 'id' column: %s", dataframe.shape)
    dataframe = lof_observation(dataframe)
    logger.info("Outlier detection and removal completed")
    visialize_data(dataframe)
    logger.info("Data visualization completed")
    PersistenceManager.save_to_pickle(dataframe)
    iterate_models(dataframe)

if __name__ == "__main__":
    main()