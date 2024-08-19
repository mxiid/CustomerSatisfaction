from cs import logger

from zenml import pipeline

from cs.components.ingest_data import ingest_data
from cs.components.clean_data import clean_data
from cs.components.train_model import train_model
from cs.components.evaluate_model import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse, mse = evaluate_model(model, X_test, y_test)
