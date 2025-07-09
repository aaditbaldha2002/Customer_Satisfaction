import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin

from src.model_dev import LinearRegressionModel
from steps.config import ModelNameConfig


@step
def model_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains a regression model on the provided data.

    Args:
        X_train: Training feature set.
        X_test: Test feature set.
        y_train: Training labels.
        y_test: Test labels.
        config: Configuration object with model parameters.

    Returns:
        A trained sklearn RegressorMixin model instance.
    """
    try:
        model = None

        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model '{config.model_name}' is not supported.")

    except Exception as e:
        logging.error(f"❌ Error in training step: {e}")
        raise
