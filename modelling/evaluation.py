import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, confusion_matrix, classification_report
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)

@with_logging
@benchmark
def evaluate_model(y_pred:np.ndarray , y_test:np.ndarray) -> np.ndarray:
    """
    Evaluate the performance of a machine learning model using various metrics.

    :param y_pred: Predicted labels from the model.
    :param y_test: True labels for the test set.
    :return: Confusion matrix as a NumPy array.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC Score: {roc_auc:.4f}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:\n%s", conf_matrix)
    print("Confusion Matrix:\n", conf_matrix)
    class_report = classification_report(y_test, y_pred)
    logger.info("Classification Report:\n%s", class_report)
    print("Classification Report:\n", class_report)
    return conf_matrix
    