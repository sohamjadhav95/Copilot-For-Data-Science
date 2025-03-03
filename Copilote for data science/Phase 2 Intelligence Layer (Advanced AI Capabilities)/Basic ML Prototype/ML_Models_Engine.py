import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error
from Data import HardcoreDataProcessor, filepath


def build_model():
    """
    Trains an ML model using AutoGluon and saves it with target column and model name.
    """
    # Preprocess data
    preprocessor = HardcoreDataProcessor(verbose=False)
    processed_file, report = preprocessor.process_dataset(filepath())
    print("Preprocessing completed. Report: \n", report)
    
    df = pd.read_csv(processed_file)
    target_column = report.get("target_column")
    task_type = report.get("task_type")

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


def test_model():
    """
    Loads the saved AutoGluon model and evaluates it on test data.
    """
    preprocessor = HardcoreDataProcessor(verbose=False)
    processed_file, report = preprocessor.process_dataset(filepath())
    
    df = pd.read_csv(processed_file)
    target_column = report.get("target_column")
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

def deploy_model():
    """
    Loads and returns the saved AutoGluon model for deployment.
    """
    preprocessor = HardcoreDataProcessor(verbose=False)
    processed_file, report = preprocessor.process_dataset(filepath())
    
    target_column = report.get("target_column")
    if not target_column:
        print("Could not determine the target column.")
        return None

    # Load model
    model_dir = f"./autogluon_model_{target_column}"
    predictor = TabularPredictor.load(model_dir)

    print(f"Model '{target_column}' is deployed and ready for inference.")
    return predictor


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
    model = deploy_model()
    if model:
        predictions = model.predict(pd.read_csv(filepath()))  # new_data should be a pandas DataFrame
        print(predictions)
