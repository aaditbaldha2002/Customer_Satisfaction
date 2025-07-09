import logging
from typing import Tuple
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreProcessStrategy
from typing_extensions import Annotated

@step
def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],

]:
    """
    Cleans the data and divides it into train and test set

    Args:
        df: Raw data
    Returns:
        X_train: Training data
        X_test: testing data
        y_train: Training labels
        y_test: testing labels
    """
    try:
        process_strategy=DataPreProcessStrategy()
        data_cleaning=DataCleaning(df,process_strategy)
        processed_data=data_cleaning.handle_data()

        data_divide_strategy=DataDivideStrategy()
        data_cleaning=DataCleaning(processed_data,data_divide_strategy)
        X_train,X_test,y_train,y_test=data_cleaning.handle_data()
        return X_train,X_test,y_train,y_test
        logging.info("Data cleaning completed")
    except Exception as e:
        logging.error(f"Error in cleaning data:{e}")
        raise e