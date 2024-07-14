from cs import logger 

import pandas as pd 
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def run(self):
        logger.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """ingests data from data_path

    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        raise e 

