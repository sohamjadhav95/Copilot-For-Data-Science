import pandas as pd
import os
import hashlib

import sys
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 2 Intelligence Layer (Advanced AI Capabilities)\v2.5 Final Touch\Core_Automation_Engine")
from Data import filepath

data = filepath()

def get_data():
    """Load, clean, update the CSV file, and return the updated file path."""
    updated_file_path = update_csv_file(data)
    return updated_file_path

def get_dataset_name():
    """Extract dataset name from file path."""
    return os.path.splitext(os.path.basename(data))[0]

def get_dataset_hash():
    """Generate a hash of the dataset to uniquely identify it."""
    df = pd.read_csv(data)
    # Create a hash based on column names and first 100 rows
    data_sample = df.head(100)
    data_str = str(list(df.columns)) + str(data_sample.values.tolist())
    return hashlib.md5(data_str.encode()).hexdigest()[:10]

def clean_numeric_column(df, column_name, remove_chars=None, unit_patterns=None):
    """
    Clean numeric columns by removing specified characters and unit patterns.
    """
    if remove_chars is None:
        remove_chars = ['$', '£', '€', ',']
    if unit_patterns is None:
        unit_patterns = [' mi.', ' km', ' miles']
    
    temp_value = df[column_name].astype(str)
    
    # Remove currency symbols and other characters
    for char in remove_chars:
        temp_value = temp_value.str.replace(char, '', regex=True)
    
    # Remove unit patterns
    for pattern in unit_patterns:
        temp_value = temp_value.str.replace(pattern, '', regex=True)
    
    df[column_name] = pd.to_numeric(temp_value, errors='coerce')

def update_csv_file(file_path):
    """Load, clean, and update the dataset CSV file."""
    df = pd.read_csv(file_path)
    
    # Identify numeric columns that might need cleaning
    for column in df.columns:
        if df[column].dtype == 'object':
            if df[column].astype(str).str.contains('[$£€]', regex=True).any():
                clean_numeric_column(df, column, remove_chars=['$', '£', '€', ','])
            elif df[column].astype(str).str.contains('(mi|mile|km)', regex=True).any():
                clean_numeric_column(df, column, unit_patterns=[' mi.', ' km', ' miles', 'mi', 'km'])
            elif df[column].astype(str).str.contains(',', regex=True).any():
                clean_numeric_column(df, column, remove_chars=[','])
    
    # Save cleaned data back to CSV
    df.to_csv("updated_file.csv", index=False)
    return file_path

def get_sample_rows(df, n=100):
    """Get first and last n rows of the dataset."""
    return df.head(n), df.tail(n)
