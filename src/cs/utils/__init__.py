from cs import logger
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class DataStrategy(ABC):
    """abstract class defining strategy for handling data"""

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """strategy for preprocessing data"""

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """preprocesses data

        Args:
            data: input data
        Returns:
            pd.DataFrame: preprocessed data
        """
        logger.info("Preprocessing data")
        # Preprocess the data
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True
            )
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logger.info(f"Error in preprocessing data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """strategy for dividing data into train and test"""

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """divides data into train and test

        Args:
            data: input data
        Returns:
            Union[pd.DataFrame, pd.Series]: train and test data
        """
        logger.info("Dividing data")
        # Divide the data
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.info(f"Error in dividing data: {e}")
            raise e


class DataCleaning:
    """class to clean data and divides it into train and test"""

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """cleans data

        Returns:
            Union[pd.DataFrame, pd.Series]: handles data
        """
        return self.strategy.handle_data(self.data)


class Model(ABC):
    """Abstract class for all models"""

    @abstractmethod
    def train(self, X_train, y_train):
        """Trains the model

        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
        """
        pass


class LinearRegressionModel(Model):

    def train(self, X_train, y_train, **kwargs):
        """Trains the model

        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
        """

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logger.info("Model trained successfully")
            return reg
        except Exception as e:
            logger.info(f"Error in training model: {e}")
            raise e


class Evaluation(ABC):
    """class to evaluate model"""

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates scores
        Args:
            y_true: np.ndarray: true labels
            y_pred: np.ndarray: predicted labels
        Returns:
            None
        """
        pass


class MSE(Evaluation):
    """class to calculate mean squared error"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates scores
        Args:
            y_true: np.ndarray: true labels
            y_pred: np.ndarray: predicted labels
        Returns:mo
            None
        """
        try:
            logger.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logger.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logger.info(f"Error in calculating scores: {e}")
            raise e


class R2(Evaluation):
    """Calculates R2 score"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates scores
        Args:
            y_true: np.ndarray: true labels
            y_pred: np.ndarray: predicted labels
        Returns:
            None
        """
        try:
            logger.info("Calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logger.info(f"R2 score: {r2}")
            return r2
        except Exception as e:
            logger.info(f"Error in calculating scores: {e}")
            raise e


class RMSE(Evaluation):
    """Calculates RMSE"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates scores
        Args:
            y_true: np.ndarray: true labels
            y_pred: np.ndarray: predicted labels
        Returns:
            None
        """
        try:
            logger.info("Calculating RMSE")
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            logger.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logger.info(f"Error in calculating scores: {e}")
            raise e
