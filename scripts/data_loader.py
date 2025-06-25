# import Python libraries
import pandas as pd
import os


# Define to manage filenames
DATA_FILES = {
    "csvpath": "data.csv"
}

# Get full path for 
def get_file_path(file_key):
    current_dir = os.getcwd()
    return os.path.join(current_dir, "../Data/raw/", DATA_FILES[file_key])


# Data loader class
class CSVDataloader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)