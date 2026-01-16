import sys
import os
import logging

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from dataclasses import dataclass

from src.utils import save_object,evaluate_model
from src.exception import CustomException

@dataclass

class ModelTrainerConfig:
    model_trainer_config_path:str = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
            
        try:
            logging.info("Split train and test data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest":RandomForestRegressor(),
                "Linear Regression":LinearRegression(),
                "SVM":SVR(),
                "KNN":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "AdaBoost":AdaBoostRegressor(),
                "XG Boost":XGBRegressor(),
                "Cat Boost":CatBoostRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
            }

            model_scores:dict = evaluate_model(X_train,y_train,X_test,y_test,models)

            best_model_score = max(sorted(model_scores.values()))

            best_model_name =  list(model_scores.keys())[
                list(model_scores.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found!")
            
            logging.info(f"Best model found: {best_model_name}")

            save_object(
                file_path = self.model_trainer_config.model_trainer_config_path,
                obj = best_model
            )

            predict = best_model.predict(X_test)

            score = r2_score(y_test,predict)

            return score

        except Exception as e:
            raise CustomException(e,sys)