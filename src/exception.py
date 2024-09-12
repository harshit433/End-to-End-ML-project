import sys

def error_message_detail(error, error_detail : sys):
    _,_,error_tb = error_detail.exc_info()
    error_message = 'error occured in python script named [{0}] line number [{1}] and error message is [{2}]'.format(error_tb.tb_frame.f_code.co_filename, error_tb.tb_lineno, error)
    return error_message
           
class CustomException(Exception):
    def __init__(self, error_message, error_detail :sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message   