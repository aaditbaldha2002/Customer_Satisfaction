from zenml import pipelines
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.model_evaluate import model_evaluate

@pipelines()
def training_pipeline(data_path:str):
    df=ingest_data(data_path)
    clean_data(df)
    model_train(df)
    model_evaluate(df)
