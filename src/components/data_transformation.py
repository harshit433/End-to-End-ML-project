import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.preprocessor_config = DataTransformationConfig()
    
    def get_data_preprocessor_object(self):
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('std_scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)    
            
    

    def initiate_data_transformation(self,train_path,test_path):

        try:
            logging.info("Entered the data transformation method")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read the dataframe successfully')

            preprocessing_obj = self.get_data_preprocessor_object()
            target_column_name = 'math_score'

            X_train = train_df.drop(columns=[target_column_name],axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name],axis=1)
            y_test = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(X_train)
            input_feature_test_arr  = preprocessing_obj.transform(X_test)

            train_arr = np.c_[
                input_feature_train_arr, np.array(y_train)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(y_test)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.preprocessor_config.preprocessor_obj_file_path,
                object = preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.preprocessor_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

