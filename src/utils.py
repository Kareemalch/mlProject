import os
import pickle
import sys
import numpy as np
import pandas as pd
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
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            try:
                # Get parameters for current model
                params = param.get(model_name, {})
                
                if params:
                    # Perform GridSearchCV if parameters are provided
                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=3,  # 3-fold cross validation
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )
                    gs.fit(X_train, y_train)
                    
                    # Use the best model for predictions
                    best_model = gs.best_estimator_
                    print(f"Best parameters for {model_name}: {gs.best_params_}")
                else:
                    # If no parameters, just fit the model as is
                    best_model = model
                    best_model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                
                # Calculate R2 scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                
                # Store the test score in report
                report[model_name] = test_model_score
                
                print(f"{model_name} - Train R2: {train_model_score:.4f}, Test R2: {test_model_score:.4f}")
                
                # Update the original model with the best parameters
                models[model_name] = best_model
                
            except Exception as model_error:
                print(f"Error training {model_name}: {str(model_error)}")
                # Set a very low score for failed models
                report[model_name] = -999.0
                continue
        
        return report
        
    except Exception as e:
        print(f"Error in evaluate_models: {str(e)}")
        raise CustomException(e, sys)  # Changed from 'raise e' to maintain consistency
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)