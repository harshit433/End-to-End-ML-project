import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifects','train.csv')
    test_data_path: str = os.path.join('artifects','test.csv')
    raw_data_path: str = os.path.join('artifects','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataframe successfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)

            logging.info("Train test split started")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info("Data ingestion completed successfully")
             
            return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    transformer = DataTransformation()
    train_data, test_data = transformer.initiate_data_transformation(train_data_path,test_data_path)
    trainer = ModelTrainer()
    r2_score = trainer.initiate_model_trainer(train_data, test_data)
    print(r2_score)



