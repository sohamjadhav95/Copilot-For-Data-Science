import pandas as pd
data = r"E:\Projects\Master Projects (Core)\AI-Powered-Tennis-Match-Outcome-Predictor\Tennis Match Prediction\Dataset\Main Dataset.csv"

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
