import os
import sys
import logging
import pickle
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            grid = GridSearchCV(model,param,cv = 3)
            grid.fit(X_train,y_train)

            model.set_params(**grid.best_params_)

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            score = r2_score(y_test,y_pred)

            report[list(models.keys())[i]] = score
            logging.info(f"Best Model Params: {grid.best_params_} ")
        return report

        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)    