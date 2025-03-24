import logging
import os
from datetime import datetime
import sys

class UnderwaterLogger:
    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # File handler
        LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
        ))
        
        # Stream handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s"
        ))
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
    
    def get_logger(self):
        return self.logger

# Initialize logger
logger = UnderwaterLogger(__name__).get_logger()

def log_image_processing(image_path, operation, status="completed"):
    """
    Specialized logging for image processing operations
    """
    logger.info(f"Image Processing | File: {image_path} | Operation: {operation} | Status: {status}")

def log_model_training(model_name, metrics):
    """
    Specialized logging for model training
    """
    logger.info(f"Model Training | Model: {model_name} | Metrics: {metrics}")

