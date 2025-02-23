import pandas as pd
import sqlite3
import numpy as np
from fuzzywuzzy import process

class EnhancedDataHandler:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.original_columns = self.df.columns.tolist()
        self.clean_columns = [col.lower().strip() for col in self.original_columns]
        self.conn = sqlite3.connect(":memory:")
        self.df.to_sql("data", self.conn, index=False, if_exists="replace")
        
    def get_data_summary(self):
        """Generate comprehensive dataset summary"""
        summary = "Dataset Structure:\n"
        summary += f"Rows: {len(self.df)}, Columns: {len(self.original_columns)}\n\nColumns:\n"
        
        for col in self.original_columns:
            dtype = str(self.df[col].dtype)
            unique = self.df[col].nunique()
            sample = ", ".join(map(str, self.df[col].dropna().head(3).tolist()))
            summary += f"- {col} ({dtype}): {unique} uniques\n  Sample: {sample}\n"
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                summary += f"  Stats: μ={self.df[col].mean():.2f}, min={self.df[col].min():.2f}, max={self.df[col].max():.2f}\n"
                hist, bins = np.histogram(self.df[col].dropna(), bins=5)
                summary += f"  Histogram: {''.join(['▇' if h > 0 else ' ' for h in hist])}\n"
        
        return summary

    def find_column_match(self, user_input):
        """Fuzzy match column names"""
        user_input = user_input.lower().strip()
        match, score = process.extractOne(user_input, self.clean_columns)
        return self.original_columns[self.clean_columns.index(match)] if score > 75 else None

    def save_dataset(self, output_file):
        """Save modified dataset"""
        self.df.to_csv(output_file, index=False)
        return f"Dataset saved to {output_file}"

    def get_columns(self):
        return self.original_columns

    def clean_data(self, clean_config):
        """Clean data based on user-defined rules"""
        for action, params in clean_config.items():
            if action == "remove_duplicates":
                subset = params.get("subset")
                self.df.drop_duplicates(subset=subset, inplace=True)
            elif action == "fill_missing":
                column = params["column"]
                method = params["method"]
                if method == "mean":
                    fill_value = self.df[column].mean()
                elif method == "median":
                    fill_value = self.df[column].median()
                elif method == "mode":
                    fill_value = self.df[column].mode()[0]
                else:
                    fill_value = None
                self.df[column].fillna(fill_value, inplace=True)
            elif action == "remove_outliers":
                column = params["column"]
                method = params["method"]
                multiplier = params.get("multiplier", 1.5)
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            elif action == "remove_columns":
                columns = params["columns"]
                self.df.drop(columns=columns, inplace=True)
            elif action == "rename_columns":
                old_names = params["old_names"]
                new_names = params["new_names"]
                self.df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
            elif action == "reorder_columns":
                new_order = params["new_order"]
                self.df = self.df[self.df.columns[new_order]]
            elif action == "add_column":
                new_col = params["new_column"]
                formula = params["formula"]
                self.df[new_col] = self.df.eval(formula)