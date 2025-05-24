import logging
import pandas as pd
from core.decorators import with_logging
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from modelling.modelling import predict_model
from modelling.evaluation import evaluate_model
from analysis.visualization import plot_confusion_matrix


logger:logging.Logger = logging.getLogger(__name__)

@with_logging
def iterate_models(
    dataframe: pd.DataFrame,target_column:str = "resultado_diagnóstico",
    gpu:bool =True
) ->None:
    """
    Iterate through various machine learning models to find the best one for the given dataset.

    :param dataframe: The input DataFrame containing features and target variable.
    :param target_column: The name of the target column in the DataFrame.
    :param gpu: Whether to use GPU for training models.
    :return: None
    """
    boost_obj: list
    if gpu:
        boost_obj =[
            XGBClassifier(tree_method="gpu_hist", gpu_id=0),
            CatBoostClassifier(task_type="GPU" , devices="0"),
            LGBMClassifier(device="gpu", gpu_platform_id=0, gpu_device_id=0)
        ]
    else:
        boost_obj = [
            XGBClassifier(),
            CatBoostClassifier(),
            LGBMClassifier()
        ]
    models:list = [
        LogisticRegression(),
        SVC(),
        RandomForestClassifier(),
        MultinomialNB(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        AdaBoostClassifier()
    ]
    models.extend(boost_obj)
    model_names:list[str] = []
    boost_models:list[bool] = []
    for model in models:
        if isinstance(model, (XGBClassifier, CatBoostClassifier, LGBMClassifier)):
            model_names.append(model.__class__.__name__)
            boost_models.append(True)
        else:
            model_names.append(model.__class__.__name__)
            boost_models.append(False)
    for model ,model_name,boost in zip(models,model_names,boost_models):
        print(f"Training {model_name}...")
        y_pred,y_test = predict_model(dataframe,model,target_column,boost)
        conf_matrix :np.ndarray = evaluate_model(y_pred,y_test)
        plot_confusion_matrix(conf_matrix, ['Maligno', 'Benigno'],model_name)