import pandas as pd
from pycaret.classification import *
from pycaret.regression import *
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *
from Core_Automation_EngineData import Data_rows, filepath
from Core_Automation_EngineNL_processor import result_response

# Rest of the code remains the same...

def build_model(user_input):
    """
    Build a machine learning model using AutoML (PyCaret).
    """
    try:
        # Load the dataset
        data_path = filepath()
        df = pd.read_csv(data_path)

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

        # Determine if the task is classification or regression
        if df[target].dtype == "object":  # Classification
            setup = setup(data=df, target=target, session_id=123)
            best_model = compare_models()
        else:  # Regression
            setup = setup(data=df, target=target, session_id=123)
            best_model = compare_models()

        # Save the best model
        model_path = f"{target}_automl_model.pkl"
        save_model(best_model, model_path)
        print(f"Best model saved to {model_path}")

        # Generate response
        result_response(user_input, f"AutoML model built and saved to {model_path}")

    except Exception as e:
        print(f"Error in build_model: {e}")