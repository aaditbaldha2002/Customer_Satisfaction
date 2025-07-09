import numpy as np
import pandas as pd
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer,)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from steps.clean_data import clean_data
from steps.config import ModelNameConfig
from steps.model_evaluate import model_evaluate
from steps.ingest_data import ingest_data
from steps.model_train import model_train
from pydantic import BaseModel

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):
    """Deployment Trigger config"""
    min_accuracy:float=0.92

@step
def deployment_trigger(
    accuracy:float,
    config:DeploymentTriggerConfig
):
    """Implements a simple model deployment trigger that looks at the input model accuracy and trigger if it the model is good enough to deploy"""
    return accuracy>=config.min_accuracy
    

@pipeline(enable_cache=True,settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    min_accuracy:float = 0.92,
    workers: int=1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_data("dataset/olist_customers_dataset.csv")
    X_train, X_test, y_train, y_test = clean_data(df)
    
    model = model_train(
        X_train, X_test, y_train, y_test,
        config=ModelNameConfig(model_name="LinearRegression")
    )

    r2_score, rmse = model_evaluate(model, X_test, y_test)
    deploy_decision=deployment_trigger(r2_score)
    mlflow_model_deployer_step(model=model,
                               deploy_decision=deploy_decision,
                               workers=workers,
                               timeout=timeout,
                               )