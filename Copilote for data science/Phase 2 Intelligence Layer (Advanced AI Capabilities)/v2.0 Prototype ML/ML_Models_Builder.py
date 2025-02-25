from groq import Groq
import pandas as pd
from pycaret.regression import setup as reg_setup, compare_models, save_model
from Data import Data_rows, filepath
from NL_processor import result_response

# Configure the Groq API
client = Groq(api_key="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")

def build_model(user_input):
    """
    Build a machine learning model using AutoML (PyCaret).
    """
    try:
        # Load the dataset
        data_path = filepath()
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully. Columns:", df.columns.tolist())

        # Extract target variable from user input
        prompt = (
            f"Extract the target variable from the following input: {user_input}\n"
            "Respond ONLY with the target variable name."
        )
        completion = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            stop=None,
        )
        target = completion.choices[0].message.content.strip()
        print("Target variable extracted:", target)

        # Check if the target variable exists and is numeric
        if target not in df.columns:
            print(f"Error: Target variable '{target}' not found in the dataset.")
            return
        if not pd.api.types.is_numeric_dtype(df[target]):
            print(f"Error: Target variable '{target}' must be numeric for regression.")
            return

        # Check for missing values in the target column
        if df[target].isnull().sum() > 0:
            print(f"Error: Target variable '{target}' has missing values. Handle them first.")
            return

        # Run PyCaret setup
        print("Setting up regression task...")
        try:
            reg_setup(data=df, target=target, session_id=123, verbose=False)
            best_model = compare_models()
        except Exception as e:
            print(f"PyCaret setup failed: {str(e)}")
            return

        # Save the model
        model_path = f"{target}_automl_model.pkl"
        save_model(best_model, model_path)
        print(f"Best model saved to {model_path}")

        # Generate response
        result_response(user_input, f"AutoML model built and saved to {model_path}")

    except Exception as e:
        print(f"Error in build_model: {str(e)}")
