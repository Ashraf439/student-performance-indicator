import os
import sys
import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    artifact_dir: str = os.path.join(os.getcwd(),"artifact")
    train_path: str = os.path.join(artifact_dir, "train.csv")
    test_path: str = os.path.join(artifact_dir, "test.csv")
    raw_path: str = os.path.join(artifact_dir, "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion method")
        try:
            df = pd.read_csv("src/notebook/data/stud.csv")
            logging.info("Dataset loaded successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_path, index=False, header=True)
            logging.info("Raw data saved")

            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)

            logging.info("Train-Test split completed")
            return self.ingestion_config.train_path, self.ingestion_config.test_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    di = DataIngestion()
    train_path, test_path = di.initiate_data_ingestion()
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")

    data_transform = DataTransformation()
    train_arr,test_arr,_ = data_transform.initiate_data_transformation(train_path,test_path)
    print("Data transform completed")

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr))