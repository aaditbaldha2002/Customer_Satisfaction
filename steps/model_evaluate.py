import logging
import pandas as pd
from zenml import step

@step
def model_evaluate(df:pd.DataFrame)->None:
    """
    Evaluates the model trained on the ingested data

    Args:
        df: dataset for evaluating the model

    Returns:
        float: accuracy value of the model on the testing dataset
    """
    pass