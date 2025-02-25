from groq import Groq
import pandas as pd
import joblib
from Data import Data_rows, filepath
from NL_processor import result_response
from ML_Model_Tester import test_model
from ML_Models_Builder import build_model

client = Groq(api_key="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")

def deploy_model(user_input):
    """
    Deploy a trained model for inference.
    """
    try:
        # Load the dataset
        data_path = filepath()
        df = pd.read_csv(data_path)

        # Extract model name from user input
        prompt = (
            f"Extract the model name from the following input: {user_input}\n"
            f"Refer to the dataset columns for context: {df.columns.tolist()}\n"
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
        model_path = f"{model_name}_model.pkl"
        model = joblib.load(model_path)

        # Generate predictions
        predictions = model.predict(df.drop(columns=[model_name]))
        print("Predictions:", predictions)

        # Generate response
        result_response(user_input, f"Predictions: {predictions}")

    except Exception as e:
        print(f"Error in deploy_model: {str(e)}")
        



def create_and_deploy_model(user_input):
    """
    Create and deploy a trained model.
    """
    try:
        build_model(user_input)
        test_model(user_input)
        deploy_model(user_input)
    
    except Exception as e:
        print(f"Error in create_and_deploy_model: {str(e)}")