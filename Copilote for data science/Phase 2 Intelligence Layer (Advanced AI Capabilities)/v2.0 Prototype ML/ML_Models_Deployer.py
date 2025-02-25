from groq import Groq
import pandas as pd
from pycaret.classification import load_model, predict_model
from pycaret.regression import load_model, predict_model
from Data import Data_rows, filepath
from NL_processor import result_response
from ML_Models_Builder import build_model

client = Groq(api_key="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")

def deploy_model(user_input):
    """
    Deploy a trained AutoML model for inference.
    """
    try:
        # Load the dataset
        data_path = filepath()
        df = pd.read_csv(data_path)

        # Extract model name from user input
        prompt = (
            f"Extract the model name from the following input: {user_input}\n"
            "Respond ONLY with the model name."
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
        model_name = completion.choices[0].message.content.strip()

        # Load the saved model
        model_path = f"{model_name}_automl_model.pkl"
        model = load_model(model_path)

        # Generate predictions
        predictions = predict_model(model, data=df)
        print("Predictions:", predictions)

        # Generate response
        result_response(user_input, f"Predictions: {predictions}")

    except Exception as e:
        print(f"Error in deploy_model: {e}")

    # Fallback as if there is no model found in directory
    create_and_deploy_model(user_input)

def create_and_deploy_model(user_input):
    """
    Create and deploy a trained AutoML model.
    """
    try:
        # Build the model
        build_model(user_input)
        
        # Deploy the model
        deploy_model(user_input)
    
    except Exception as e:
        print(f"Error in create_and_deploy_model: {e}")