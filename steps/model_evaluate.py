import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from src.evaluation import MSE, RMSE, R2
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow
import mlflow.sklearn

# Use active experiment tracker (optional if using MLflow for logging)
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_evaluate(
    model_uri: str,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
    Evaluates a logged MLflow model on test data and logs metrics.

    Args:
        model_uri: URI to the MLflow model.
        X_test: Feature test set.
        y_test: Ground truth labels for test set.

    Returns:
        r2_score: Coefficient of determination.
        rmse: Root mean squared error.
    """
    try:
        # ✅ Load model from MLflow
        model = mlflow.sklearn.load_model(model_uri)
        predictions = model.predict(X_test)

        # ✅ Calculate metrics
        mse = MSE().calculate_scores(y_test, predictions)
        r2 = R2().calculate_scores(y_test, predictions)
        rmse = RMSE().calculate_scores(y_test, predictions)

        # ✅ Log metrics to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)

        logging.info(f"✅ Evaluation complete — R²: {r2:.4f}, RMSE: {rmse:.4f}")
        return r2, rmse

    except Exception as e:
        logging.error(f"❌ Error in model evaluation: {e}")
        raise
