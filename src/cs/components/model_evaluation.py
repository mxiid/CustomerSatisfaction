from cs import logger

from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """Evaluates the model

    Args:
        df: input data
    """
    logger.info("Evaluating model")
    # Evaluate the model
    pass