import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from pyarrow import binary
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error
from Preprocessing_whole_data import DatasetPreprocessor
import pandas as pd
from groq import Groq


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

import os
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from Preprocessing_whole_data import DatasetPreprocessor


def deploy_model(user_input):
    """
    Deploys the trained model, makes predictions, and visualizes results with actual values.
    """
    # Preprocess data
    preprocessor = DatasetPreprocessor(verbose=False)
    processed_file, report = preprocessor.process_dataset()
    df = pd.read_csv(processed_file)

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

    # Display actual vs predicted values if actual values exist
    if target_column in df.columns:
        comparison_df = df[[target_column]].copy()
        comparison_df["Predicted"] = predictions
        print("\nActual vs Predicted Values:")
        print(comparison_df.head(20))  # Display first 20 results

        # Display predictions with some features
        display_df = df.iloc[:20].copy()  # Take first 20 rows
        display_df["Predicted"] = predictions[:20]
        print("\nPredictions with Some Features:")
        print(display_df.head(20))

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(df[target_column], predictions, alpha=0.5, label="Predicted vs Actual", color="blue")
        plt.plot([df[target_column].min(), df[target_column].max()], 
                 [df[target_column].min(), df[target_column].max()], 'r', lw=2)  # Reference line
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("\nPredicted Values:")
        print(predictions.head(20))  # Show only predictions if actual values are unavailable


if __name__ == "__main__":
    deploy_model("Housing")