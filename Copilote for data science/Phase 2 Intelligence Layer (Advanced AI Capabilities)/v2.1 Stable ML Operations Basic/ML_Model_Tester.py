from groq import Groq
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, mean_squared_error
from Data import Data_rows, filepath
from NL_processor import result_response

client = Groq(api_key="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")

def test_model(user_input):
    """
    Test the performance of a trained model.
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
        model_path = f"{model_name}_model.pkl"
        model = joblib.load(model_path)

        # Generate predictions
        X = df.drop(columns=[model_name])
        y_true = df[model_name]
        y_pred = model.predict(X)

        # Evaluate the model
        if df[model_name].dtype == "object":  # Classification
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Model accuracy: {accuracy}")
            result_response(user_input, f"Model accuracy: {accuracy}")
        else:  # Regression
            mse = mean_squared_error(y_true, y_pred)
            print(f"Model MSE: {mse}")
            result_response(user_input, f"Model MSE: {mse}")

    except Exception as e:
        print(f"Error in test_model: {str(e)}")