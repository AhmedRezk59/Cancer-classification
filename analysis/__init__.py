import logging
import pandas as pd
from analysis.analysis import analyze_dataframe, find_missing_values
from analysis.visualization import plot_count, plot_heatmap, plot_scatter, plot_distribution, boxplot_dist

logger: logging.Logger = logging.getLogger(__name__)

def numerical_eda(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    EDA based on numerical values for dataset
    :param dataframe: Dataframe to analyze
    :type dataframe: pd.DataFrame
    :return: The dataframe without missing values
    :rtype: pd.DataFrame
    """
    analyze_dataframe(dataframe)
    dataframe = find_missing_values(dataframe)
    return dataframe

def visialize_data(dataframe: pd.DataFrame) -> None:
    """
    Visualize the data using various plots.
    :param dataframe: The input dataframe.
    :type dataframe: pd.DataFrame
    :return: None
    """
    logger.info("Running visualization")
    plot_heatmap(dataframe)  # area & perimetro, compacidad & simetria
    plot_scatter(dataframe, 'área', 'perímetro', 'resultado_diagnóstico')
    plot_scatter(dataframe, 'compacidad', 'simetría', 'resultado_diagnóstico')
    plot_count(
        dataframe, ['radio', 'textura', 'perímetro', 'área', 'suavidad',
                    'compacidad', 'simetría', 'dimensión_fractal'],
        'resultado_diagnóstico')
    plot_distribution(dataframe['resultado_diagnóstico'], 'lightskyblue')
    boxplot_dist(dataframe, 'textura', 'perímetro')
