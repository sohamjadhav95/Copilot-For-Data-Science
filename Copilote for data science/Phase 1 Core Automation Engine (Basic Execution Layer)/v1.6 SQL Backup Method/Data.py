import pandas as pd
data = r"C:\Users\soham\Downloads\synthetic_sales_data.csv"

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
        