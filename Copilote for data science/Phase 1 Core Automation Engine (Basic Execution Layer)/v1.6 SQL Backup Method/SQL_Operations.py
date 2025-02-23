import pandas as pd
import sqlite3
from Data import filepath  # Import the filepath function

class SQLExecutor:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')  # Create an in-memory SQLite database
        self.table_name = self._get_table_name()  # Dynamically get the table name from the file path
        self._load_data()  # Load the CSV data into the database

    def _get_table_name(self):
        """Extract the table name from the file path (without extension)"""
        data_path = filepath()  # Use the filepath function to get the file path
        table_name = data_path.split("\\")[-1].split(".")[0]  # Extract file name without extension
        return table_name

    def _load_data(self):
        """Load CSV data into the in-memory SQL database"""
        data_path = filepath()  # Use the filepath function to get the file path
        df = pd.read_csv(data_path)  # Read the CSV file into a DataFrame
        # Load the DataFrame into the SQL database with the dynamically generated table name
        df.to_sql(self.table_name, self.conn, index=False, if_exists='replace')

    def execute_sql(self, query):
        """Execute SQL query and return results"""
        try:
            result = pd.read_sql_query(query, self.conn)
            return result, True
        except Exception as e:
            print(f"SQL Error: {e}")
            return None, False

    def save_changes(self):
        """Save modified data back to CSV"""
        data_path = filepath()  # Use the filepath function to get the file path
        updated_df = pd.read_sql(f"SELECT * FROM {self.table_name}", self.conn)
        updated_df.to_csv(data_path, index=False)