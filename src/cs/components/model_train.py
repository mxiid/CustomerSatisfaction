import mlflow
import mlflow.sklearn
from cs import logger 
import pandas as pd
from zenml import step

from cs.utils import LinearRegressionModel
from sklearn.base import RegressorMixin
from cs.config import ModelNameConfig

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig) -> RegressorMixin:
    """Trains the model

    Args:
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.DataFrame
        y_test: pd.DataFrame
    """
    logger.info("Training model")
    # Train the model
    try:
        model = None 
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logger.info(f"Error in training model: {e}")
        raise e