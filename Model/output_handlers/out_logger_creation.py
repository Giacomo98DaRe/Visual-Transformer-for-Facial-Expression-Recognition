# IMPORT

import logging
import os

########################################################################################################################
# LOGGING

def logger_creation(run_path, run_counter):
    epoc_acc_logging_path = os.path.join(run_path, f"epoch_acc_log_{run_counter}.log")

    # First logger
    logger1 = logging.getLogger('Epoch_Acc_logger')
    logger1.setLevel(logging.INFO)
    file_handler1 = logging.FileHandler(epoc_acc_logging_path)
    logger1.addHandler(file_handler1)

    return logger1
