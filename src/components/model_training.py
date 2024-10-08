import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifects", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info("Spliting training and test data")
            X_train, y_train,X_test,y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict =evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models)

            best_model_name = max(model_report, key=model_report.get)   

            best_model = models[best_model_name]

            if model_report[best_model_name] < 0.6:
                raise CustomException("Model performance is not good")    
            
            logging.info("Best model found on both training and testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2
        
        
        except Exception as e:
            raise CustomException(e, sys)