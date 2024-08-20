from cs import logger

from zenml import pipeline

from cs.components.ingest_data import ingest_data
from cs.components.clean_data import clean_data
from cs.components.model_train import train_model
from cs.components.model_evaluation import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    mse, r2, rmse = evaluate_model(model, X_test, y_test)
