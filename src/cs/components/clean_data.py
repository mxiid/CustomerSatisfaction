from cs import logger
import pandas as pd
from zenml import step
from cs.utils import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"], 
    Annotated[pd.DataFrame, "X_test"], 
    Annotated[pd.Series, "y_train"], 
    Annotated[pd.Series, "y_test"]
    ]:
    """Cleans the data

    Args:
        df: input data
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: train and test data
    """
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logger.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.info (f"Error in cleaning data: {e}")
        raise e