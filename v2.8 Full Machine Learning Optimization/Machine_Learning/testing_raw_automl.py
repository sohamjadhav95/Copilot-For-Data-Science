import pandas as pd
import os
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
import joblib
from pycaret.clustering import *
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 2 Intelligence Layer (Advanced AI Capabilities)\v2.8 Full Machine Learning Optimization\Core_Automation_Engine")
from Data import filepath


client = Groq(api_key="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")

def targeted_column(user_input):
    df = pd.read_csv(filepath())

    prompt = (
        f"Get the target column name for Machine Learning from user input: {user_input}.\n"
        f"Refer this available dataset columns: {df.columns.tolist()}\n"
        f"Respond 'ONLY With Target Column Name' as in the dataset\n"
        f"If target column name is not mentioned in user input or does not match dataset columns, "
        f"then determine the target column by referring the dataset: {df.head(200)}"
    )

    try:
        completion = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            stop=None,
        )
        response = completion.choices[0].message.content.strip()

        if response not in df.columns:
            raise ValueError(f"Invalid target column received: {response}")

        target_column = response
    except Exception as e:
        print(f"Error in fetching target column: {e}")
        target_column = "target"  # Default fallback

    print(f"Target column: {target_column}")
    return target_column

def task_type():
    """
    Determines the machine learning task type based on the dataset and user input.
    Returns: 'Regression', 'Binary Classification', 'Multiclass Classification', or 'Clustering'
    """
    df = pd.read_csv(filepath())
    dtype = df.dtypes

    # Create prompt for Groq API
    prompt = (
        f"Determine the machine learning task type based on this information:\n"
        f"Data types: {dtype}\n"
        f"Sample dataset: {df.head(100)}\n\n"
        f"Choose ONLY ONE from these options:\n"
        f"- Regression: if target is continuous numeric\n"
        f"- Binary Classification: if target has exactly 2 unique values\n"
        f"- Multiclass Classification: if target has more than 2 unique categorical values\n"
        f"- Clustering: if no clear target column exists\n"
        f"Respond ONLY with the chosen task type."
    )
    

    completion = client.chat.completions.create(
        model="qwen-2.5-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=50,
        top_p=0.9,
        stream=False,
        stop=None,
    )
    task = completion.choices[0].message.content.strip()
    print(f"Task type: {task}")
    
    # Validate response
    valid_tasks = ['Regression', 'Binary Classification', 'Multiclass Classification', 'Clustering']
    if task not in valid_tasks:
        raise ValueError(f"Invalid task type received: {task}")
        
    return task

if __name__ == "__main__":
    print(task_type())