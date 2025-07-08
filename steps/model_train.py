import logging
import pandas as pd
from zenml import step

@step
def model_train(df:pd.DataFrame)->None:
    """
    Trains the model on the ingested data

    Args:
        df:the ingested data
    Retuns:
        None
    """
    pass