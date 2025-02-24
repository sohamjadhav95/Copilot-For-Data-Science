import pandas as pd
from pycaret.classification import load_model, predict_model
from pycaret.regression import load_model, predict_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from Core_Automation_Engine.Data import Data_rows, filepath
from Core_Automation_Engine.NL_processor import result_response

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
        target = model._final_estimator.target
        y_true = df[target]
        y_pred = predictions["Label"]

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
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            result_response(user_input, f"Model performance: MSE={mse}, MAE={mae}")

    except Exception as e:
        print(f"Error in test_model: {e}")