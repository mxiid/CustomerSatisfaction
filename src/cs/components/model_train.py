from cs import logger 
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    """Trains the model

    Args:
        df: input data
    """
    logger.info("Training model")
    # Train the model
    pass