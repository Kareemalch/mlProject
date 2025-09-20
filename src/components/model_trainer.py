import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42, eval_metric='rmse'),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }
            
            params = {
                "Decision Tree": {
                    'max_depth': [3, 5, 7, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'max_depth': [3, 4, 5, 6, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                },
                "XGBRegressor": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.01, 0.1, 1],
                    'reg_lambda': [0, 0.01, 0.1, 1],
                    'min_child_weight': [1, 3, 5, 7]
                },
                "CatBoosting Regressor": {
                    'iterations': [50, 100, 200, 300],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128, 255],
                    'bagging_temperature': [0, 0.5, 1.0]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0, 1.5],
                    'loss': ['linear', 'square', 'exponential']
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models, 
                param=params
            )

            # Get best model
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.warning(f"Best model score {best_model_score} is below 0.6 threshold")

            logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make final predictions and calculate R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            logging.info(f"Final R2 score on test set: {r2_square}")
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)