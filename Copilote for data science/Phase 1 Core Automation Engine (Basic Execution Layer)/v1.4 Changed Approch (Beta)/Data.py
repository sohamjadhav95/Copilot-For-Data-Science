import pandas as pd

def Data_rows():
    try:
        data = r"C:\Users\soham\Downloads\Financial Sample.csv"
        dataset = pd.read_csv(data)
        first_100_rows = dataset.head(100)
        last_100_rows = dataset.tail(100)
        return first_100_rows, last_100_rows, data
    except Exception as e:
        print(f"An error occurred in Data_rows: {e}")
        return None, None
        