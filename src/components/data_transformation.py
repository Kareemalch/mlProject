import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object




@dataclass
class DataTransformationConfig():
    data_preprocessor = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:  
    def __init__(self):   
        self.data_transform_config = DataTransformationConfig()
    
    def data_transformer(self):
        try:
            numerical_columns = [ 'reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'Categorical columns: {categorical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ('numerical', numerical_pipeline, numerical_columns),
                    ("categorical", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    

    def start_data_transformation (self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("train and test data completed")

            preprocessing_obj = self.data_transformer()
            
            numerical_columns = [ 'reading_score', 'writing_score']
            target_column = 'math_score'

            input_feature_train_df= train_df.drop(columns=[target_column],axis=1)
            out_feature_train_df = train_df[target_column]


            input_feature_test_df= test_df.drop(columns=[target_column],axis=1)
            out_feature_test_df = test_df[target_column]

            logging.info("applying preprocessing on train and test data")


            in_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            in_test_arr = preprocessing_obj.transform(input_feature_test_df)

            save_object(

                file_path=self.data_transform_config.data_preprocessor,
                obj=preprocessing_obj
            )

            train_arr = np.c_[
                in_train_arr, np.array(out_feature_train_df)  
            ]

            test_arr = np.c_[
                in_test_arr, np.array(out_feature_test_df)   
            ]

            return train_arr,test_arr,self.data_transform_config.data_preprocessor
        except Exception as e:
            raise CustomException(e,sys)