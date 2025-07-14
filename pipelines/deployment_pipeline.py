import json
import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError
from sklearn.base import RegressorMixin

from zenml import  step, pipeline
from zenml.services import ServiceConfig, ServiceType
from zenml.logger import get_logger
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer)
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from steps.clean_data import clean_data
from steps.config import ModelNameConfig
from steps.model_evaluate import model_evaluate
from steps.ingest_data import ingest_data
from steps.model_train import model_train

# ✅ Logger setup
logger = get_logger(__name__)

# ✅ Docker requirements for MLflow
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# ✅ Configuration for deployment trigger
class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.0

# ✅ Trigger step to decide deployment
@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
) -> bool:
    """Trigger model deployment if accuracy meets the defined threshold."""
    should_deploy = accuracy >= config.min_accuracy
    logger.info("Deployment decision: %s (accuracy: %.4f, threshold: %.4f)", should_deploy, accuracy, config.min_accuracy)
    return should_deploy

# ✅ Deployment step using MLflow URI
@step
def custom_mlflow_model_deployer_step(
    model_uri: str,  # ✅ corrected type
    deploy_decision: bool,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
) -> Optional[MLFlowDeploymentService]:
    """Deploys the model using MLflow if deployment is triggered."""
    if not deploy_decision:
        logger.info("🚫 Deployment skipped: Accuracy below threshold.")
        return None

    if not model_uri:
        raise ValueError("❌ Model URI is empty. Ensure model was logged correctly.")

    deployer = MLFlowModelDeployer.get_active_model_deployer()

    deployment_config_dict = {
        "model_name": "bankruptcy-risk-predictor",
        "pipeline_name": "deploy_pipeline",
        "port": 5000,
        "replicas": 1,
        "environment": {"ENV": "production", "TZ": "UTC"},
        "resources": {"cpu": "1", "memory": "512Mi"},
        "model_uri": model_uri,
    }

    try:
        deployment_config = ServiceConfig(**deployment_config_dict)
    except ValidationError as e:
        print("Validation error in ServiceConfig:", e)
        raise
    
    service = deployer.deploy_model(
        deployment_config,
        service_type = ServiceType(type="rest-api", flavor="default"),
        replace=True,  # replace existing deployment if needed
        continuous_deployment_mode=True,  # enable continuous deployment mode
        timeout=timeout,
    )
    logger.info("✅ Model deployed successfully at: %s", service.prediction_url)
    return service

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


# ✅ Continuous deployment pipeline
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path:str,
    min_accuracy: float = 0.0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    """Pipeline for ingesting, training, evaluating, and deploying a model."""
    # Step 1: Ingest & clean data
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)

    # Step 2: Train model and log to MLflow
    model = model_train(
        X_train, X_test, y_train, y_test,
        config=ModelNameConfig(model_name="LinearRegression")
    )

    # Step 3: Evaluate the model
    r2_score, rmse = model_evaluate(model, X_test, y_test)

    # Step 4: Decide whether to deploy
    deploy_decision = deployment_trigger(
        accuracy=r2_score,
        config=DeploymentTriggerConfig(min_accuracy=min_accuracy)
    )

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy_decision,
        workers=workers,
        timeout=timeout
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)
