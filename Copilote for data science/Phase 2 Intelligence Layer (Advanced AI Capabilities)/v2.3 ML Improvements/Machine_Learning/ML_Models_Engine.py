import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from pyarrow import binary
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error
from Preprocessing_whole_data import DatasetPreprocessor
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
import joblib

import sys
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 2 Intelligence Layer (Advanced AI Capabilities)\v2.3 ML Improvements\Core_Automation_Engine")
from Data import filepath


client = Groq(api_key="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")


def targeted_column(user_input):
    # Preprocess data
    preprocessor = DatasetPreprocessor(verbose=False)
    processed_file, report = preprocessor.process_dataset()
    df = pd.read_csv(processed_file)

    prompt = (
            f"Get the target column name for Machine Learning from user input: {user_input}.\n"
            f"Refer this avilable dataset columns: {df.columns.tolist()}\n"
            f"Respond 'ONLY With Target Column Name' as in the dataset\n"
            f"If target column name is not mentioned in user input or in not match with dataset columns, then determine the target column by referring the dataset: {df.head(200)}"
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

    if report.get("task_type") == "clustering":
        print("Clustering task detected. No target column required.")
        target_column = None
    else:
        target_column = response
        print(f"Target column: {target_column}")
        return target_column


def build_model(user_input):
    """
    Trains an ML model using AutoGluon and saves it with target column and model name.
    """
    # Preprocess data
    preprocessor = DatasetPreprocessor(verbose=False)
    processed_file, report = preprocessor.process_dataset()

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
    # Preprocess data
    preprocessor = DatasetPreprocessor(verbose=False)
    processed_file, report = preprocessor.process_dataset()

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
    Deploys the trained model, makes predictions, and visualizes results with actual values.
    """
    # Get raw and processed data
    preprocessor = DatasetPreprocessor(verbose=False)
    processed_file, report = preprocessor.process_dataset()
    raw_file = filepath()  # Get raw dataset file path
    
    df = pd.read_csv(processed_file)  # Processed Data
    raw_df = pd.read_csv(raw_file)    # Raw Data (before preprocessing)

    target_column = targeted_column(user_input)
    if not target_column:
        print("Could not determine the target column.")
        return None

    # Define model directory
    model_dir = f"./autogluon_model_{target_column}"

    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"Model '{target_column}' not found. Building the model...")
        build_model(user_input)

    # Load the trained model
    predictor = TabularPredictor.load(model_dir)
    print(f"Model '{target_column}' is deployed and ready for inference.")

    # Prepare test data
    X_test = df.drop(columns=[target_column]) if target_column in df.columns else df
    predictions = predictor.predict(X_test)

    # Load original (raw) actual values
    raw_actual_values = raw_df[target_column] if target_column in raw_df.columns else df[target_column]

    ## ---- ✅ FIX: APPLY INVERSE TRANSFORMATION CORRECTLY ---- ##
    scaler_path = "scalers.pkl"
    if os.path.exists(scaler_path):
        scalers = joblib.load(scaler_path)

        if target_column in scalers:
            scaler = scalers[target_column]
            predictions_original = scaler.inverse_transform(predictions.values.reshape(-1, 1)).flatten()
        else:
            print(f"Warning: No scaler found for {target_column}. Using processed predictions.")
            predictions_original = predictions
    else:
        print("Warning: No scalers file found. Using processed predictions.")
        predictions_original = predictions

    # Display actual vs predicted values in original format
    if target_column in df.columns:
        comparison_df = df[[target_column]].copy()
        comparison_df["Predicted"] = predictions
        comparison_df["Actual_Original"] = raw_actual_values
        comparison_df["Predicted_Original"] = predictions_original  # Decoded values

        print("\nActual vs Predicted Values (Processed Data):")
        print(comparison_df[[target_column, "Predicted"]].head(20))

        print("\nActual vs Predicted Values (Original Format):")
        print(comparison_df[["Actual_Original", "Predicted_Original"]].head(20))

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(raw_actual_values, predictions_original, alpha=0.5, label="Predicted vs Actual", color="blue")
        plt.plot([raw_actual_values.min(), raw_actual_values.max()], 
                 [raw_actual_values.min(), raw_actual_values.max()], 'r', lw=2)  # Reference line
        plt.xlabel("Actual Values (Original)")
        plt.ylabel("Predicted Values (Original)")
        plt.title("Actual vs Predicted Values (Original Data)")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("\nPredicted Values (Processed Data):")
        print(predictions.head(20))


if __name__ == "__main__":
    deploy_model("Predict by ML Model")