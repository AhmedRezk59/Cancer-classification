import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.neighbors import LocalOutlierFactor
from engineering.persistence_manager import DataType
from core.decorators import with_logging , benchmark


@with_logging
@benchmark
def lof_observation(
    dataframe: pd.DataFrame,
    data_type:DataType = DataType.FIGURES
) -> pd.DataFrame:
    """
    Apply Local Outlier Factor (LOF) to the dataframe to detect and remove outliers.

    :param dataframe: The input dataframe.
    :param data_type: The type of data (default is DataType.F).
    :return: The dataframe with outliers removed.
    """
    df_num_cols = dataframe.select_dtypes(include=np.number)
    df_outlier:pd.DataFrame = df_num_cols.astype(float)
    clf :LocalOutlierFactor = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.1
    )
    clf.fit_predict(df_outlier)
    df_scores = clf.negative_outlier_factor_
    scores_df:pd.DataFrame = pd.DataFrame(np.sort(df_scores))
    scores_df.plot(
        stacked=True,xlim=[0,20],color="red",title='Visualization of outliers according to the LOF method', style='.-')
    import os
    output_path = f'{data_type.value}outliers.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    th_val = np.sort(df_scores)[2]
    outliers = df_scores < th_val
    print(dataframe.shape)
    dataframe :pd.DataFrame= dataframe.drop(df_outlier[outliers].index).reset_index()
    print(dataframe.shape)
    return dataframe


@with_logging
@benchmark
def random_oversample(x_train, y_train):
    """
    Random oversample the minority class using RandomOverSampler from
    the imbalanced-learn library.
    :param x_train: Feature matrix of the training set
    :type x_train: numpy.ndarray or pandas.DataFrame
    :param y_train: Target vector of the training set
    :type y_train: numpy.ndarray or pandas.Series
    :return: Random oversampled feature matrix and target vector
    :rtype: numpy.ndarray or pandas.DataFrame, numpy.ndarray or pandas.Series
    """
    random_over_sampler: RandomOverSampler = RandomOverSampler(random_state=42)
    x_ros, y_ros = random_over_sampler.fit_resample(x_train, y_train)
    return x_ros, y_ros


