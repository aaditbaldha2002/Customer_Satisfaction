from typing import cast
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
import click 
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer)
from zenml.integrations.mlflow.services import MLFlowDeploymentService


DEPLOY="deploy"
PREDICT="predict"
DEPLOY_AND_PREDICT="deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY,PREDICT,DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optianally you can choose to only run the deployment"
    "pipeline to train and deploy a model (`deploy`), or to"
    "only run a prediction against the deployed model"
    "(`predict`). By default both will be run"
    "(`deploy_and_predict`),",
)
@click.option("--min-accuracy",default=0,help="Minimum accuracy required to deploy the model",)

def run_deployment(config:str, min_accuracy:float):
    mlflow_model_deployer_component=MLFlowModelDeployer.get_active_model_deployer()
    deploy=config==DEPLOY or config==DEPLOY_AND_PREDICT
    predict=config==PREDICT or config==DEPLOY_AND_PREDICT

    if deploy:
        continuous_deployment_pipeline(data_path="dataset/olist_customers_dataset.csv",min_accuracy=min_accuracy,workers=3,timeout=60)
    if predict:
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step"
        )

    print(
        "You can run:\n"
        f"[italic green] mlflow ui --backend-store-uri {get_tracking_uri()}"
        "[/italic green]\n ... to inspect your experiment runs within the MLflow"
        "UI.\n You can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to "
        "compare two or more runs.\n\n"
    )

    existing_services=mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuos_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service=cast(MLFlowDeploymentService,existing_services[0])
        if service.is_running:
            print(
                f"The MLFlow prediction server is running locally as a daemon "
                f"process service and accents inference, requests at:\n"
                f"       {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green] `zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]"
            )
        elif service.is_failed:
            print(
                f"The MLFlow prediction server is in a failed state:\n"
                f"Last state: '{service.status.state.value}'\n"
                f"Last error: '{service.status.last_error}'"
            )
        else:
            print("No MLFlow prediction service is currently running. The deployment"
                  "pipeline must run first to train a model and deploy it. Execute "
                  "the same command with that `--deploy` argument to deploy a model.")
    
if __name__=='__main__':
    run_deployment()
