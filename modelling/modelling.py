import logging
import numpy as np
import pandas as pd
from core.decorators import with_logging, benchmark
from modelling.preprocessing import random_oversample
from modelling.train import training

logger: logging.Logger = logging.getLogger(__name__)

@with_logging
@benchmark
def predict_model(
    dataframe:pd.DataFrame,
    ml_model,
    target_column:str,
    boost:bool = False,
) -> tuple:
    x_train , x_test , y_train , y_test = training(
        dataframe,
        target_column
    )
    x_train, y_train = random_oversample(
        x_train,
        y_train
    )
    if boost:
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
    ml_model.fit(x_train, y_train)
    y_pred :np.ndarray = ml_model.predict(x_test)
    logger.info(x_train.shape , y_train.shape)
    return y_pred , y_test