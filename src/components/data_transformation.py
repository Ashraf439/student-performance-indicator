import sys
import os
import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            categorical_column = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course']
            numerical_column = ['math_score', 'writing_score']

            numeric_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),# fills the missing values with median
                    ('Standard Scaler',StandardScaler())
                ]
            )
            logging.info(f"Numerical columns: {numerical_column}")
            category_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),# Fills the missing values with mode
                    ('One Hot',OneHotEncoder())
                ]
            )
            logging.info(f"Categorical columns: {categorical_column}")

            preprocessor = ColumnTransformer(
                [
                    ('numeric pipeline',numeric_pipeline,numerical_column),
                    ('category pipeline',category_pipeline,categorical_column)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            target_feature = 'reading_score'
            logging.info("Read the train & test data")
            preprocessing_obj = self.get_data_transformer_object()

            X_train = train_df.drop(columns=[target_feature],axis=1)
            y_train = train_df[target_feature]

            X_test = test_df.drop(columns=[target_feature],axis=1)
            y_test = test_df[target_feature]

            logging.info("Applying the preprocessing on the train and test data")

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr,np.array(y_train)]
            test_arr = np.c_[X_test_arr,np.array(y_test)]


            save_object(
                file_path = self.data_tranformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Saved preprocessing object")

            return (train_arr,test_arr,self.data_tranformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)


