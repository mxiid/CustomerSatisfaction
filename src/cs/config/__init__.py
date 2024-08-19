from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """model configs
    """
    model_name: str = "LinearRegression"