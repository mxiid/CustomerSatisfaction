from cs import logger

import pandas as pd
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step

from cs.utils import MSE, RMSE, R2 

@step
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[Annotated[float, "mse"], Annotated[float, "r2"], Annotated[float, "rmse"]]:
    """Evaluates the model

    Args:
        model: RegressorMixin: trained model
        X_test: pd.DataFrame: test data
        y_test: pd.DataFrame: test labels
    Returns:
        Tuple[float, float, float]: mse, r2, rmse
    """

    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        return mse, r2, rmse
    except Exception as e:
        logger.info(f"Error in evaluating model: {e}")
        raise e