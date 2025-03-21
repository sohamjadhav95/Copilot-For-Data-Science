import pandas as pd
import numpy as np
import io
import json
from groq import Groq
import warnings
import os

warnings.filterwarnings("ignore")


def identify_problem_type(data, target_column, client):
    """
    Identify the type of machine learning problem (classification, regression, or clustering)
    using both LLM-based and statistical approaches.
    """
    # If no target column is specified, try LLM approach first
    if target_column is None or target_column not in data.columns:
        result = _get_llm_prediction(data.head(100), client)
        if result:
            llm_target_column, task_type = result
            if task_type in ['regression', 'classification']:
                # If LLM identified a valid target column, use it for statistical verification
                if llm_target_column in data.columns:
                    statistical_type = detect_task_type_statistically(data[llm_target_column])
                    # If LLM and statistical approaches agree, return that type
                    if statistical_type == task_type:
                        return task_type
                    # Otherwise, trust the statistical approach more
                    return statistical_type
                return task_type
        return 'clustering'

    # If target column exists, use statistical approach
    if target_column in data.columns:
        return detect_task_type_statistically(data[target_column])
    
    return 'unknown'

def _get_llm_prediction(sample_data, client):
    """
    Use LLM to identify the target column and task type.
    """
    buffer = io.StringIO()
    
    buffer.write(f"Dataset Shape: {sample_data.shape}\n\n")
    buffer.write("Column Names and Types:\n")
    for col, dtype in sample_data.dtypes.items():
        buffer.write(f"- {col}: {dtype}\n")
    
    buffer.write("\nColumn Statistics:\n")
    for col in sample_data.columns:
        if not pd.api.types.is_numeric_dtype(sample_data[col]):
            unique_vals = sample_data[col].nunique()
            buffer.write(f"- {col}: {unique_vals} unique values (non-numeric)\n")
            if unique_vals < 10:
                buffer.write(f"  Sample values: {', '.join(map(str, sample_data[col].dropna().unique()[:5]))}\n")
        else:
            stats_data = sample_data[col].describe()
            buffer.write(f"- {col}: min={stats_data['min']:.2f}, max={stats_data['max']:.2f}, "
                        f"mean={stats_data['mean']:.2f}, std={stats_data['std']:.2f}, "
                        f"unique values={sample_data[col].nunique()}\n")
    
    buffer.write("\nFirst 100 rows:\n")
    buffer.write(sample_data.head(100).to_string(index=False))

    dataset_summary = buffer.getvalue()
    
    prompt = f"""
    You are an expert data scientist. Analyze this dataset and determine:
    1. Which column is most likely the target variable for a machine learning task?
    2. Is this a regression task or a classification task?
    
    Rules:
    - If the target column contains mostly continuous numbers, it is a regression task.
    - If the target column has a small set of discrete values (e.g., categories), it is a classification task.
    
    Dataset information:
    {dataset_summary}
    
    Provide your analysis in JSON format only:
    {{
        "target_column": "name_of_target_column",
        "task_type": "classification_or_regression"
    }}
    """

    # print(f"Dataset Summary : {dataset_summary}")
    
    try:
        completion = client.chat.completions.create(
            model="qwen-2.5-coder-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
            top_p=0.9,
            stream=False,
        )
        
        response = completion.choices[0].message.content.strip().lower()
        result = json.loads(response)
        
        if "target_column" in result and "task_type" in result:
            return result["target_column"], result["task_type"]
    except Exception:
        pass  # Suppress errors
    
    return None, None  # Default if LLM fails

def detect_task_type_statistically(column):
    """
    Statistically determine if a column is better suited for regression or classification.
    """
    if not pd.api.types.is_numeric_dtype(column):
        return 'classification'
        
    unique_ratio = column.nunique() / len(column)
    
    # Check if values are mostly integers or have very few decimal places
    is_mostly_integer = column.dtype == np.int64 or (column.round() == column).mean() > 0.95
    
    # More sophisticated classification detection
    if (unique_ratio < 0.05 and column.nunique() < 10) or \
       (is_mostly_integer and column.nunique() < 10 and column.max() - column.min() < 10):
        return 'classification'
    
    # Check distribution characteristics for regression
    # Continuous data often has a more normal-like distribution
    try:
        # Check skewness and kurtosis for signs of continuous distribution
        skewness = abs(stats.skew(column.dropna()))
        kurtosis = stats.kurtosis(column.dropna())
        
        # If data has reasonable skewness and many unique values, likely regression
        if column.nunique() > 20 and unique_ratio > 0.1:
            return 'regression'
    except:
        pass
    
    # If values are spread across a wide range with many unique values, likely regression
    if unique_ratio > 0.2 or column.nunique() > 30:
        return 'regression'
    
    # Default based on unique values and range
    if column.nunique() <= 15:
        return 'classification'
    
    return 'regression'

# Example usage
if __name__ == "__main__":
    data = get_data()
    client = Groq(api_key="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")
    target_column = "Price"  # Default target column

    problem_type = identify_problem_type(data, target_column, client)
    print(f"Problem Type: {problem_type}")