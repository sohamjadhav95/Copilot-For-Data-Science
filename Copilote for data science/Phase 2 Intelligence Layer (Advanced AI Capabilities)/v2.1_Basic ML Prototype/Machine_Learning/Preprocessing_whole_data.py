import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from datetime import datetime
import warnings
import argparse
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
import requests  # Add this
import json     # Add this
import io       # Add this
from scipy import stats  # Add this
from groq import Groq
from data import get_data
from problem_identifier import identify_problem_type


class DatasetPreprocessor:
    """
    A class to preprocess any dataset (CSV) for machine learning tasks
    including regression, classification, or clustering.
    """
    
    def __init__(self, verbose=True):
        """Initialize the preprocessor with options."""
        self.verbose = verbose
        self.categorical_columns = []
        self.numerical_columns = []
        self.date_columns = []
        self.target_column = None
        self.task_type = None  # 'regression', 'classification', or 'clustering'
        warnings.filterwarnings('ignore')
        # Initialize Groq client with API key
        self.client = Groq(api_key="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")  # Replace with your actual Groq API key

    def print_info(self, message):
        """Print information if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def select_csv_file(self):
        """
        Open a file dialog to select a CSV file.
        Returns the file path or None if canceled.
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        root.destroy()
        return file_path if file_path else None
    
    def select_output_directory(self):
        """
        Open a directory dialog to select where to save the output.
        Returns directory path or None if canceled.
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        dir_path = filedialog.askdirectory(
            title="Select Output Directory"
        )
        
        root.destroy()
        return dir_path if dir_path else None
    
    def load_data(self, file_path=None):
        """Load CSV data into a pandas DataFrame."""
        # If no file_path provided, prompt user to select one
        if file_path is None:
            file_path = self.select_csv_file()
            if file_path is None:
                print("No file selected. Exiting.")
                return False
        
        self.print_info(f"Loading data from {file_path}...")
        try:
            self.df = pd.read_csv(file_path)
            self.original_df = self.df.copy()
            self.file_name = os.path.basename(file_path)
            self.print_info(f"Successfully loaded data with shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def inspect_data(self):
        """Inspect the dataset to understand its structure."""
        self.print_info("\n=== Dataset Information ===")
        
        # Display basic info
        self.print_info(f"Number of rows: {self.df.shape[0]}")
        self.print_info(f"Number of columns: {self.df.shape[1]}")
        
        # Display column types
        self.print_info("\n=== Column Data Types ===")
        self.print_info(self.df.dtypes)
        
        # Missing values
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        missing_info = pd.DataFrame({
            'Missing Values': missing_values,
            'Missing Percent': missing_percent
        })
        self.print_info("\n=== Missing Values ===")
        self.print_info(missing_info[missing_info['Missing Values'] > 0])
        
        # Statistical summary
        self.print_info("\n=== Numerical Statistics ===")
        self.print_info(self.df.describe().T)
        
        # Sample data
        self.print_info("\n=== Sample Data ===")
        self.print_info(self.df.head())
        
        return self.df
    
    def identify_column_types(self):
        """Identify numerical, categorical, and date columns."""
        # First pass - use data types
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numerical_columns.append(col)
            else:
                # Check if column might be a date
                if self._is_date_column(col):
                    self.date_columns.append(col)
                else:
                    self.categorical_columns.append(col)
        
        # Second pass - check if numeric columns might actually be categorical
        for col in list(self.numerical_columns):
            if len(self.df[col].unique()) < 10:  # Threshold for considering a numeric col as categorical
                self.numerical_columns.remove(col)
                self.categorical_columns.append(col)
        
        self.print_info(f"\nNumerical columns: {self.numerical_columns}")
        self.print_info(f"Categorical columns: {self.categorical_columns}")
        self.print_info(f"Date columns: {self.date_columns}")
    
    def _is_date_column(self, column):
        """Check if a column contains date values."""
        # Sample a few values to check if they can be converted to dates
        sample = self.df[column].dropna().head(5)
        date_count = 0
        
        for val in sample:
            try:
                pd.to_datetime(val)
                date_count += 1
            except:
                pass
        
        # If most of the samples could be converted to dates, consider it a date column
        return date_count > len(sample) / 2


    def _get_llm_prediction(self, sample_data):
        """
        Use an LLM to identify the target column and task type.
        
        Args:
            sample_data: A sample of the dataset (first 100 rows)
            
        Returns:
            tuple: (target_column, task_type)
        """
        # Create a summary of the dataset for the LLM
        buffer = io.StringIO()
        
        # Write basic dataset info
        buffer.write(f"Dataset Shape: {sample_data.shape}\n\n")
        buffer.write("Column Names and Types:\n")
        for col, dtype in sample_data.dtypes.items():
            buffer.write(f"- {col}: {dtype}\n")
        
        buffer.write("\nColumn Statistics:\n")
        for col in sample_data.columns:
            # Skip if column has non-numeric data
            if not pd.api.types.is_numeric_dtype(sample_data[col]):
                unique_vals = sample_data[col].nunique()
                buffer.write(f"- {col}: {unique_vals} unique values (non-numeric)\n")
                # Show sample values for categorical columns
                if unique_vals < 10:
                    buffer.write(f"  Sample values: {', '.join(map(str, sample_data[col].dropna().unique()[:5]))}\n")
            else:
                # For numeric columns, show basic stats
                stats_data = sample_data[col].describe()
                buffer.write(f"- {col}: min={stats_data['min']:.2f}, max={stats_data['max']:.2f}, "
                           f"mean={stats_data['mean']:.2f}, std={stats_data['std']:.2f}, "
                           f"unique values={sample_data[col].nunique()}\n")
        
        buffer.write("\nSample Data (first 5 rows):\n")
        buffer.write(sample_data.head().to_string())
        
        dataset_summary = buffer.getvalue()
        print(f"Dataset Summary: {dataset_summary}")
        
        # Prepare the prompt for the LLM
        prompt = f"""
        You are an expert data scientist. Analyze this dataset and determine:
        1. Which column is most likely the target variable for a machine learning task?
        2. Is this a regression task or a classification task?
        
        For classification tasks, the target typically has categorical values or a small number of distinct values.
        For regression tasks, the target is usually continuous with many possible values.
        
        Dataset information:
        {dataset_summary}
        
        Provide your analysis in JSON format only:
        {{
            "target_column": "name_of_target_column",
            "task_type": "classification_or_regression",
            "confidence": 0-100,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self._call_llm_api(prompt)
            result = json.loads(response)
            
            self.print_info(f"\nLLM Analysis Result:")
            self.print_info(f"Target Column: {result['target_column']}")
            # self.print_info(f"Task Type: {result['task_type']}")
            self.print_info(f"Task Type: {identify_problem_type(self.df, self.target_column, self.client)}")
            self.print_info(f"Confidence: {result['confidence']}%")
            self.print_info(f"Reasoning: {result['reasoning']}")
            # self.task_type = result['task_type']
            self.task_type = identify_problem_type(self.df, self.target_column, self.client)    # Saving Task type
            
            return result['target_column'], identify_problem_type(self.df, self.target_column, self.client)
        except Exception as e:
            self.print_info(f"Error using LLM for target detection: {str(e)}")
            self.print_info("Falling back to heuristic methods")
            return None, None

    def _call_llm_api(self, prompt):
        """
        Call the Groq LLM API service.
        
        Args:
            prompt: The text prompt for the LLM
            
        Returns:
            str: The LLM's response
        """
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1000,
                top_p=0.95,
                stream=False,
                stop=None,
            )

            return completion.choices[0].message.content.strip().lower()
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")

    def detect_task_type_statistically(self, column):
        """
        Statistically determine if a column is better suited for regression or classification.
        """
        # Skip if non-numeric
        if not pd.api.types.is_numeric_dtype(column):
            return 'classification'
            
        # If very few unique values compared to total size, likely classification
        unique_ratio = column.nunique() / len(column)
        if unique_ratio < 0.05:  # Less than 5% of values are unique
            return 'classification'
            
        # Check if values are mostly integers or round numbers
        if column.dtype == np.int64 or (column.round() == column).mean() > 0.9:
            # If small range of integers, likely classification
            if column.max() - column.min() < 10:
                return 'classification'
        
        # Test for normality
        try:
            if column.nunique() <= 1:
                return 'classification'
                
            _, p_value = stats.normaltest(column.dropna())
            if p_value > 0.05:  # Not enough evidence to reject normality
                return 'regression'
        except:
            pass
        
        # Check if values cluster around specific points
        try:
            kde = stats.gaussian_kde(column.dropna())
            x = np.linspace(column.min(), column.max(), 100)
            density = kde(x)
            peaks = sum(1 for i in range(1, len(density)-1) 
                      if density[i-1] < density[i] > density[i+1])
            
            if peaks > 2:
                return 'classification'
        except:
            pass
            
        return 'regression'
    
    def detect_ml_task(self):
        """
        Automatically detect if the dataset is suited for regression, classification, or clustering.
        Uses LLM if available, falls back to heuristic method.
        """
        self.print_info("\n=== Detecting Machine Learning Task Type ===")
        
        # First try with LLM
        sample_data = self.df.head(100)  # Send first 100 rows to LLM
        llm_target, llm_task = self._get_llm_prediction(sample_data)

        self.print_info(f"DF columns: {self.df.columns}")
        
        if llm_target.upper() and llm_target.upper() in self.df.columns:
            self.target_column = llm_target
            self.task_type = identify_problem_type(self.df, self.target_column, self.client)
            self.print_info(f"Using LLM-detected target: '{self.target_column}' for {self.task_type}")
            return self.target_column
        
        # Fall back to heuristic method if LLM fails
        self.print_info("Using heuristic target detection...")
        
        # Rest of the existing detect_ml_task method...
        # [Keep the existing code for the heuristic approach]
        
    def set_target_column(self, target_col=None):
        """
        Set the target column for supervised learning tasks.
        If not specified, try to detect it automatically.
        """
        # First determine the task type
        if target_col and target_col in self.df.columns:
            self.target_column = target_col
            self.print_info(f"Target column set to: {self.target_column}")
        else:
            # No target specified, try to detect automatically
            self.target_column = self.detect_ml_task()
            self.task_type = identify_problem_type(self.df, self.target_column, self.client)  # Ensure task_type is set

        # If task type is clustering, set target_column to None
        if self.task_type == 'clustering':
            self.target_column = None
            self.print_info("Task type is clustering (unsupervised learning). Target column set to None.")
        

        self.print_info(f"Task type set to: {self.task_type}")

    
    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        self.print_info("\n=== Handling Missing Values ===")
        
        # For numerical columns, use mean imputation
        if self.numerical_columns:
            num_imputer = SimpleImputer(strategy='mean')
            self.df[self.numerical_columns] = num_imputer.fit_transform(self.df[self.numerical_columns])
            self.print_info("Imputed missing numerical values with mean")
            
        # For categorical columns, use mode imputation
        if self.categorical_columns:
            cat_cols_with_missing = [col for col in self.categorical_columns 
                                    if self.df[col].isnull().sum() > 0 and col in self.df.columns]
            
            for col in cat_cols_with_missing:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            
            if cat_cols_with_missing:
                self.print_info(f"Imputed missing categorical values with mode")
    
    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(self.df)
        
        self.print_info(f"\n=== Removed {removed} duplicate rows ===")
    
    def handle_outliers(self):
        """
        Detect and handle outliers in numerical columns using the IQR method.
        """
        self.print_info("\n=== Handling Outliers ===")
        
        for col in self.numerical_columns:
            # Skip if target column or if column was removed
            if col == self.target_column or col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Cap outliers
                self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                self.print_info(f"Capped {outliers} outliers in column '{col}'")
    
    def encode_categorical_features(self):
        """
        Encode categorical variables:
        - One-hot encoding for nominal data (with few categories)
        - Label encoding for ordinal data or high-cardinality features
        """
        self.print_info("\n=== Encoding Categorical Features ===")
        
        for col in self.categorical_columns:
            # Skip target column if it's a classification task
            if col == self.target_column and self.task_type == 'classification':
                # For classification target, use label encoding
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.print_info(f"Label encoded target column '{col}'")
                continue
                
            # Skip if column not in dataframe (might have been removed)
            if col not in self.df.columns:
                continue
                
            # If column has few categories (nominal data), use one-hot encoding
            if len(self.df[col].unique()) <= 10:
                # Get dummies and preserve the original dataframe's index
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(col, axis=1)
                self.print_info(f"One-hot encoded column '{col}' ({len(dummies.columns)} new features)")
            else:
                # For high-cardinality features, use label encoding
                le = LabelEncoder()
                self.df[f"{col}_encoded"] = le.fit_transform(self.df[col].astype(str))
                self.df = self.df.drop(col, axis=1)
                self.print_info(f"Label encoded column '{col}' (high cardinality)")
    
    def normalize_numerical_features(self):
        """Normalize numerical features using Min-Max scaling."""
        self.print_info("\n=== Normalizing Numerical Features ===")
        
        # Don't normalize the target for regression
        num_cols = [col for col in self.numerical_columns 
                   if (self.task_type != 'regression' or col != self.target_column) 
                   and col in self.df.columns]
        
        if num_cols:
            scaler = MinMaxScaler()
            self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
            self.print_info(f"Normalized {len(num_cols)} numerical features")
            
    def get_target_and_task_type(self):
        """
        Returns the target column and task type identified for the dataset.
        
        Returns:
            tuple: (target_column, task_type) where:
                - target_column is the name of the identified target variable (or None for clustering)
                - task_type is one of 'regression', 'classification', or 'clustering'
        """
        return self.target_column, self.task_type
    
    def perform_feature_engineering(self):
        """Create new features from existing ones."""
        self.print_info("\n=== Performing Feature Engineering ===")
        
        # Extract features from date columns
        for col in self.date_columns:
            if col in self.df.columns:
                try:
                    # Convert to datetime
                    self.df[col] = pd.to_datetime(self.df[col])
                    
                    # Extract useful components
                    self.df[f'{col}_year'] = self.df[col].dt.year
                    self.df[f'{col}_month'] = self.df[col].dt.month
                    self.df[f'{col}_day'] = self.df[col].dt.day
                    self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
                    
                    # Add to numerical columns
                    self.numerical_columns.extend([f'{col}_year', f'{col}_month', 
                                                 f'{col}_day', f'{col}_dayofweek'])
                    
                    # Drop the original column
                    self.df = self.df.drop(col, axis=1)
                    self.print_info(f"Extracted date features from '{col}'")
                except:
                    self.print_info(f"Could not convert '{col}' to datetime")
        
        # Calculate interaction terms between numerical features
        num_cols = [col for col in self.numerical_columns 
                   if col != self.target_column and col in self.df.columns]
        
        # Limit to prevent too many features
        if len(num_cols) >= 2 and len(num_cols) <= 5:
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    col1, col2 = num_cols[i], num_cols[j]
                    interaction_col = f"{col1}_x_{col2}"
                    self.df[interaction_col] = self.df[col1] * self.df[col2]
                    self.numerical_columns.append(interaction_col)
                    self.print_info(f"Created interaction feature: {interaction_col}")
    
    def handle_imbalanced_data(self):
        """
        Handle imbalanced data using SMOTE for classification tasks.
        Only apply if we have a target column and it's a classification problem.
        """
        if not self.target_column or self.task_type != 'classification':
            self.print_info("\n=== Skipping imbalanced data handling (not a classification task) ===")
            return
            
        self.print_info("\n=== Handling Imbalanced Data ===")
        
        # Count class distribution
        class_counts = self.df[self.target_column].value_counts()
        self.print_info("Class distribution before balancing:")
        self.print_info(class_counts)
        
        # Check if imbalanced
        if len(class_counts) > 1 and (class_counts.max() / class_counts.min()) > 1.5:
            try:
                # Prepare feature matrix and target vector
                X = self.df.drop(self.target_column, axis=1)
                y = self.df[self.target_column]
                
                # Apply SMOTE
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                # Reconstruct dataframe
                self.df = pd.DataFrame(X_resampled, columns=X.columns)
                self.df[self.target_column] = y_resampled
                
                # Show new distribution
                new_counts = self.df[self.target_column].value_counts()
                self.print_info("Class distribution after balancing:")
                self.print_info(new_counts)
            except Exception as e:
                self.print_info(f"Could not apply SMOTE: {str(e)}")
        else:
            self.print_info("Data is already reasonably balanced. Skipping SMOTE.")
    
    def save_processed_data(self, output_path=None):
        """Save the processed dataset to a new CSV file."""
        if output_path is None:
            # Get output directory from user if not specified
            output_dir = self.select_output_directory()
            
            if output_dir:
                # Create filename based on original file and task type
                original_name = os.path.splitext(self.file_name)[0]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(
                    output_dir, 
                    f"{original_name}_{self.task_type}_processed_{timestamp}.csv"
                )
            else:
                # If user cancels directory selection, use current directory
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"processed_dataset_{timestamp}.csv"
        
        self.df.to_csv(output_path, index=False)
        self.print_info(f"\n=== Processed data saved to {output_path} ===")
        
        # Create metadata file with preprocessing information
        metadata_path = output_path.replace('.csv', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Original dataset: {self.file_name}\n")
            f.write(f"Original shape: {self.original_df.shape}\n")
            f.write(f"Processed shape: {self.df.shape}\n")
            f.write(f"Task type: {self.task_type}\n")
            # if self.target_column:
            f.write(f"Target column: {self.target_column}\n")
            f.write(f"Numerical columns: {', '.join(self.numerical_columns)}\n")
            f.write(f"Categorical columns: {', '.join(self.categorical_columns)}\n")
            f.write(f"Date columns: {', '.join(self.date_columns)}\n")
            f.write(f"Preprocessing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.print_info(f"Preprocessing metadata saved to {metadata_path}")
        
        return output_path
    
    def generate_preprocessing_report(self):
        """Generate a report of the preprocessing steps."""
        self.print_info("\n=== Preprocessing Report ===")
        self.print_info(f"Original dataset shape: {self.original_df.shape}")
        self.print_info(f"Processed dataset shape: {self.df.shape}")
        self.print_info(f"Task type: {self.task_type}")
        
        # Always show target column for supervised tasks
        if self.task_type in ['regression', 'classification']:
            self.print_info(f"Target column: {self.target_column}")
        elif self.task_type == 'clustering':
            self.print_info("Clustering task (no target column)")
        
        # Calculate feature increase/decrease
        orig_cols = self.original_df.shape[1]
        new_cols = self.df.shape[1]
        change = new_cols - orig_cols
        if change > 0:
            self.print_info(f"Added {change} new features")
        else:
            self.print_info(f"Removed {abs(change)} features")
            
        return {
            "original_shape": self.original_df.shape,
            "processed_shape": self.df.shape,
            "feature_change": change,
            "task_type": self.task_type,
            "target_column": self.target_column
        }
    
    def process_dataset(self, target_col=None):
        """Run the complete preprocessing pipeline."""
        if not self.load_data(get_data()):
            return False
        
        self.inspect_data()
        self.identify_column_types()
        self.set_target_column(target_col)
        self.handle_missing_values()
        self.remove_duplicates()
        self.handle_outliers()
        self.encode_categorical_features()
        self.normalize_numerical_features()
        self.perform_feature_engineering()
        self.handle_imbalanced_data()
        output_file = self.save_processed_data("processed_data.csv")
        report = self.generate_preprocessing_report()
        
        return output_file, report


def main():
    """
    Main function to run the preprocessor with direct file path input.
    """
    try:
        # Example usage
        file_path = get_data() # Replace with your actual file path

         # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]


        preprocessor = DatasetPreprocessor()
        result = preprocessor.process_dataset()
        
        if result:
            output_file, report = result
            print(f"\nPreprocessing completed successfully. Processed data saved to: {output_file}")
        else:
            print("\nPreprocessing failed.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
