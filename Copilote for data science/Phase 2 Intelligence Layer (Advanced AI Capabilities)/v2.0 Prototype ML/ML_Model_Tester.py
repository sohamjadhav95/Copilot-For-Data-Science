from groq import Groq
import pandas as pd
from pycaret.classification import load_model, predict_model
from pycaret.regression import load_model, predict_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from Data import Data_rows, filepath
from NL_processor import result_response

client = Groq(api_key="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")

def test_model(user_input):
    """
    Test the performance of a trained AutoML model.
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

        # Evaluate the model
        target = user_input.split()[-1]  # Extract the target variable from user input
        if target not in df.columns:
            print(f"Error: Target variable '{target}' not found in the dataset.")
            return
        y_true = df[target]
        # Ensure predictions contain the correct column name
        if target in predictions.columns:
            y_pred = predictions[target]  # Use target variable name for regression
        else:
            print(f"Error: Prediction column '{target}' not found in output.")
            return

        if df[target].dtype == "object":  # Classification
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="weighted")
            recall = recall_score(y_true, y_pred, average="weighted")
            f1 = f1_score(y_true, y_pred, average="weighted")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1-Score: {f1}")
            result_response(user_input, f"Model performance: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-Score={f1}")
        else:  # Regression
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            accuracy = r2 * 100  # R-squared as percentage accuracy
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"R2 Score: {r2}")
            print(f"Model Accuracy: {accuracy:.2f}%")
            result_response(user_input, f"Model performance: MSE={mse}, MAE={mae}")

    except Exception as e:
        print(f"Error in test_model: {e}")