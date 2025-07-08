import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from src.evaluation import MSE,RMSE,R2
from typing import Tuple
from typing_extensions import Annotated

@step
def model_evaluate(model: RegressorMixin,X_test:pd.DataFrame,y_test:pd.DataFrame)->Tuple[Annotated[float,"r2_score"],Annotated[float,"rmse"]]:
    """
    Evaluates the model trained on the ingested data

    Args:
        df: dataset for evaluating the model

    Returns:
        float: accuracy value of the model on the testing dataset
    """
    try:
        prediction=model.predict(X_test)
        mse_class=MSE()
        mse=mse_class.calculate_scores(y_test,prediction)

        r2_class=R2()
        r2=r2_class.calculate_scores(y_test,prediction)

        rmse_class=RMSE()
        rmse=rmse_class.calculate_scores(y_test,prediction)

        return r2,rmse
    except Exception as e:
        logging.error("Error in model evaluation:{}".format(e))
        raise e
    