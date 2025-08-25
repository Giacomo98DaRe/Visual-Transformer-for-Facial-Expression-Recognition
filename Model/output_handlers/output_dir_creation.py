# IMPORT

import os

########################################################################################################################
# OUTPUT DIRECTORY CREATION

def out_dir_creation():

    run_counter = 1
    while True:
        folder_run_name = f"run{run_counter}"
        run_path = os.path.join('runs', folder_run_name)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            break
        run_counter += 1

    return run_path, run_counter
