from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.model_evaluate import model_evaluate

@pipeline(enable_cache=False)
def training_pipeline(data_path:str):
    df=ingest_data(data_path)
    X_train,X_test,y_train,y_test=clean_data(df)
    model=model_train(X_train,X_test,y_train,y_test)
    r2_score,rmse=model_evaluate(model,X_test,y_test)
