import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error
from Data import HardcoreDataProcessor, filepath
import pandas as pd
from groq import Groq

client = Groq(api_key="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")

# Preprocess data
preprocessor = HardcoreDataProcessor(verbose=False)
processed_file, report = preprocessor.process_dataset(filepath())

def targeted_column(user_input):
    df = pd.read_csv(processed_file)

    prompt = (
            f"Get the target column name for Machine Learning from user input: {user_input}.\n"
            f"Refer this avilable dataset columns: {df.columns.tolist()}\n"
            f"Respond 'ONLY With Target Column Name' as in the dataset\n"
            f"If target column name is not mentioned in user input or in not match with dataset columns then return no_target_column"
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

    response = completion.choices[0].message.content.strip()
    print(response)

    if response == "no_target_column":
        target_column = report.get("target_column")
    else:
        target_column = response

    return target_column


def build_model(user_input):
    """
    Trains an ML model using AutoGluon and saves it with target column and model name.
    """
    df = pd.read_csv(processed_file)
    task_type = report.get("task_type")
    target_column = targeted_column(user_input) 

    if not target_column or not task_type:
        print("Could not determine the target column or task type.")
        return None

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Define model save path
    model_dir = f"./autogluon_model_{target_column}"
    os.makedirs(model_dir, exist_ok=True)

    # Train AutoGluon model
    predictor = TabularPredictor(label=target_column, path=model_dir).fit(X_train, time_limit=300)

    print(f"Model saved at: {model_dir}")
    return predictor


def test_model(user_input):
    """
    Loads the saved AutoGluon model and evaluates it on test data.
    """
    df = pd.read_csv(processed_file)
    target_column = targeted_column(user_input) 
    task_type = report.get("task_type")
    
    if not target_column or not task_type:
        print("Could not determine the target column or task type.")
        return None

    _, X_test = train_test_split(df, test_size=0.2, random_state=42)

    # Load saved model
    model_dir = f"./autogluon_model_{target_column}"
    predictor = TabularPredictor.load(model_dir)

    # Make predictions
    y_test = X_test[target_column]
    X_test = X_test.drop(columns=[target_column])

    y_pred = predictor.predict(X_test)

    # Evaluate model
    if task_type == "classification":
        accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage
        print(f"Model Accuracy: {accuracy:.2f}%")
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) * 100  # Convert R² score to percentage
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Lower is better

        print(f"Model Mean Squared Error (MSE): {mse:.4f}")
        print(f"Model Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Model Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Model R² Score (Explained Variance): {r2:.2f}%")
        print(f"Model Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

def deploy_model(user_input):
    """
    Loads and returns the saved AutoGluon model for deployment.
    """

    df = pd.read_csv(processed_file)

    target_column = targeted_column(user_input)
    if not target_column:
        print("Could not determine the target column.")
        return None

    # Load model
    model_dir = f"./autogluon_model_{target_column}"
    predictor = TabularPredictor.load(model_dir)

    print(f"Model '{target_column}' is deployed and ready for inference.")
    model = predictor
    if model:
        predictions = model.predict(df)  # new_data should be a pandas DataFrame
        print(predictions)


def execute_ml_task():
    """
    Processes NL commands and executes ML tasks accordingly.
    """
    command = "test_model"  # Change this to "test_model" or "deploy_model" to test different functionalities.

    if command == "build_model":
        print("Building ML model...")
        build_model()
    elif command == "test_model":
        print("Testing ML model...")
        test_model()
    elif command == "deploy_model":
        print("Deploying ML model...")
        deploy_model()
    else:
        print("No valid ML task found in the input.")


if __name__ == "__main__":
    test_model("predict the outcome from the dataset") 

