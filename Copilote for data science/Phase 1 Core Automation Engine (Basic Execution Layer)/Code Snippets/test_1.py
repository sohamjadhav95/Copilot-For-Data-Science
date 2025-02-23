import pandas as pd

data_paths = [
    r"C:\Users\soham\Downloads\synthetic_sales_data.csv",
    r"C:\Users\soham\Downloads\Financial Sample.csv"
]

def Data_rows():
    try:
        datasets = [pd.read_csv(file) for file in data_paths]
        first_last_rows = [(df.head(100), df.tail(100)) for df in datasets]
        return (f"Datasets:\n{first_last_rows}")
    except Exception as e:
        print(f"An error occurred in Data_rows: {e}")
        return None

result = Data_rows()
print(result)

def filepath():
    data_path = data_paths
    return data_path
