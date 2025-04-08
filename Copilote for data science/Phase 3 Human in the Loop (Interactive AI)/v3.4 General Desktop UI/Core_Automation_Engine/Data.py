import pandas as pd
import os

# Default data path
data = r"C:\Users\soham\Downloads\iris_synthetic_data.csv"

def import_data(file_path):
    """Import data from a file path and update the global data variable"""
    global data
    try:
        if os.path.exists(file_path) and file_path.endswith('.csv'):
            # Test read the file to ensure it's valid
            pd.read_csv(file_path)
            data = file_path
            return True, f"Successfully imported data from {os.path.basename(file_path)}"
        else:
            return False, "Invalid file path or file is not a CSV"
    except Exception as e:
        return False, f"Error importing data: {str(e)}"

def update_data_path(new_path):
    """Update the data file path"""
    return import_data(new_path)

def Data_rows():
    try:
        dataset = pd.read_csv(data)
        first_100_rows = dataset.head(100)
        last_100_rows = dataset.tail(100)
        return first_100_rows, last_100_rows
    except Exception as e:
        print(f"An error occurred in Data_rows: {e}")
        return None, None

def filepath():
    data_path = data
    return data_path
        
def dataset_features():
    data_path = filepath()
    dataset = pd.read_csv(data_path)
    
    dataset_features = {
        "shape": dataset.shape,
        "size": dataset.size,
        "columns": dataset.columns,
        "dtypes": dataset.dtypes
    }
    return dataset_features
