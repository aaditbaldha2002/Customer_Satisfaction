import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin

from src.model_dev import LinearRegressionModel
from steps.config import ModelNameConfig
from zenml.client import Client
import mlflow
import mlflow.sklearn

# Fetch experiment tracker from ZenML's active stack
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains a regression model and logs it to MLflow.

    Returns:
        model_uri (str): URI of the logged model in MLflow.
    """
    try:
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)

            # ✅ Rely on ZenML-managed run
            mlflow.sklearn.log_model(trained_model, artifact_path="model")
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

            logging.info(f"✅ Model logged to MLflow with URI: {model_uri}")
            return trained_model
        else:
            raise ValueError(f"Model '{config.model_name}' is not supported.")
    except Exception as e:
        logging.error(f"❌ Error in model_train step: {e}")
        raise