import sys
import logging
from logger import logger

class UnderwaterProcessingError(Exception):
    """Base class for underwater image processing exceptions"""
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = self.error_message_detail(error_message, error_detail)
        logger.error(self.error_message)
    
    def error_message_detail(self, error, error_detail):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in python script [{file_name}] line [{line_number}] error: {str(error)}"
        return error_message
    
    def __str__(self):
        return self.error_message

class ImageQualityPredictionError(UnderwaterProcessingError):
    """Exception for image quality prediction failures"""
    def __init__(self, error_message, error_detail:sys):
        super().__init__(f"Quality Prediction Error: {error_message}", error_detail)

class ModelTrainingError(UnderwaterProcessingError):
    """Exception for model training failures"""
    def __init__(self, error_message, error_detail:sys):
        super().__init__(f"Training Error: {error_message}", error_detail)

class FeatureExtractionError(UnderwaterProcessingError):
    """Exception for feature extraction failures"""
    def __init__(self, error_message, error_detail:sys):
        super().__init__(f"Feature Extraction Error: {error_message}", error_detail)
def error_message_detail(error, error_detail:sys) :
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message="Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    
    return error_message
class CustomException(Exception) :
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message= error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message

