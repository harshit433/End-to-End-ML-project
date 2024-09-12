import numpy as np
import pandas as pd
import dill
import sys
import os
from src.exception import CustomException
from sklearn.metrics import r2_score


## SAVE A FILE 
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e,sys)


## LOAD A FILE
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)


## EVALUATE MULTIPLE MODELS TO GET THE BEST MODEL
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_report[model_name] = r2_score(y_test, y_pred)
        return model_report
    except Exception as e:
        raise CustomException(e,sys)