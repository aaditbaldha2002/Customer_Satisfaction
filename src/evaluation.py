import logging
from abc import ABC,abstractmethod
import numpy as np
from sklearn.base import r2_score
from sklearn.metrics import mean_squared_error, root_mean_squared_error

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation of our models
    """
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true:True labels
            y_pred: Values predicted by the model
        Returns:
            None
        """
        pass


class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculating MSE")
            mse=mean_squared_error(y_true,y_pred)
            logging.info("MSE:{}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation Strategy that uses R2 score
    """

    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculating the R2 score")
            r2=r2_score(y_true,y_pred)
            logging.info("R2 Score:{}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 score:{}".format(e))
            raise e

class RMSE(Evaluation):
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculating RMSE")
            rmse=root_mean_squared_error(y_true,y_pred)
            logging.info("RMSES:{}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE:{}".format(e))
            raise e