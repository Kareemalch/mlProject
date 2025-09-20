import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
class DataIngestion:
    def __init__(self):
        # Simple paths instead of dataclass
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'data.csv')
    
    def initiate_data_ingestion(self):
        logging.info("IN DATA INGESTION")
        
        try:
            # Load data
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')
            
            # Create artifacts folder
            os.makedirs('artifacts', exist_ok=True)
            
            # Split data
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=24)
            
            # Save files
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)
            
            # Return paths
            return self.train_data_path, self.test_data_path
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)

# Usage
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    # Fix: Unpack 3 values instead of 2
    train_arr, test_arr, preprocessor_path = data_transformation.start_data_transformation(train_data, test_data)
    
    modeltrainer = ModelTrainer()
    # Fix: Pass all 3 required parameters
    model_score = modeltrainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
    print(f"Model R2 Score: {model_score}")