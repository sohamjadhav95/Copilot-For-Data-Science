import pandas as pd
import os
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
import joblib
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 2 Intelligence Layer (Advanced AI Capabilities)\v2.8 Full Machine Learning Optimization\Core_Automation_Engine")
from Data import filepath


client = Groq(api_key="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")

def targeted_column(user_input):
    df = pd.read_csv(filepath())

    prompt = (
        f"Get the target column name for Machine Learning from user input: {user_input}.\n"
        f"Refer this available dataset columns: {df.columns.tolist()}\n"
        f"Respond 'ONLY With Target Column Name' as in the dataset\n"
        f"If target column name is not mentioned in user input or does not match dataset columns, "
        f"then determine the target column by referring the dataset: {df.head(200)}"
    )

    try:
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

        if response not in df.columns:
            raise ValueError(f"Invalid target column received: {response}")

        target_column = response
    except Exception as e:
        print(f"Error in fetching target column: {e}")
        target_column = "target"  # Default fallback

    print(f"Target column: {target_column}")
    return target_column

def task_type(user_input):
    """
    Determines the machine learning task type based on the dataset and user input.
    Returns: 'Regression', 'Binary Classification', 'Multiclass Classification', or 'Clustering'
    """
    df = pd.read_csv(filepath())
    target_col = targeted_column(user_input)
    
    # Analyze target column characteristics
    unique_values = df[target_col].nunique()
    dtype = df[target_col].dtype
    
    # Create prompt for Groq API
    prompt = (
        f"Determine the machine learning task type based on this information:\n"
        f"Target column: {target_col}\n"
        f"Data type: {dtype}\n"
        f"Number of unique values: {unique_values}\n"
        f"Sample values: {df[target_col].head(10).tolist()}\n\n"
        f"Choose ONLY ONE from these options:\n"
        f"- 'Regression' if target is continuous numeric\n"
        f"- 'Binary Classification' if target has exactly 2 unique values\n"
        f"- 'Multiclass Classification' if target has more than 2 unique categorical values\n"
        f"- 'Clustering' if no clear target column exists\n"
        f"Respond ONLY with the chosen task type."
    )
    
    try:
        completion = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=50,
            top_p=0.9,
            stream=False,
            stop=None,
        )
        task = completion.choices[0].message.content.strip()
        
        # Validate response
        valid_tasks = ['Regression', 'Binary Classification', 'Multiclass Classification', 'Clustering']
        if task not in valid_tasks:
            raise ValueError(f"Invalid task type received: {task}")
            
        return task
        
    except Exception as e:
        print(f"Error determining task type: {e}")
        # Fallback logic based on data characteristics
        if dtype in [np.float64, np.int64] and unique_values > 10:
            return 'Regression'
        elif unique_values == 2:
            return 'Binary Classification'
        elif unique_values > 2 and unique_values <= 10:
            return 'Multiclass Classification'
        else:
            return 'Clustering'
        
#======================================================================================


def build_model_and_test(user_input):
    """
    Trains an ML model using AutoGluon if it does not exist and always performs testing.
    """
    # Universal Parameters
    DATA_PATH = filepath()
    target_column = targeted_column(user_input)
    MODEL_DIR = f"./autogluon_model_{target_column}"
    LABEL_COLUMN = target_column
    ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

    # Load dataset
    task = task_type(user_input)
    print(f"Task Type: {task}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset:\n{df.head()}")

    # Check if model already exists
    if os.path.exists(MODEL_DIR):
        print(f"Model found at {MODEL_DIR}. Skipping training and proceeding with testing.")
        predictor = TabularPredictor.load(MODEL_DIR)

        # Load label encoder if applicable
        if os.path.exists(ENCODER_PATH):
            label_encoder = joblib.load(ENCODER_PATH)
            if df[LABEL_COLUMN].dtype == 'object':
                df[LABEL_COLUMN] = label_encoder.transform(df[LABEL_COLUMN])
    else:
        print(f"Training new model and saving to {MODEL_DIR}.")
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Encode categorical target variable if needed
        if df[LABEL_COLUMN].dtype == 'object':
            label_encoder = LabelEncoder()
            df[LABEL_COLUMN] = label_encoder.fit_transform(df[LABEL_COLUMN])
            joblib.dump(label_encoder, ENCODER_PATH)  # ✅ Save encoder

        # Train Model
        predictor = TabularPredictor(label=LABEL_COLUMN, path=MODEL_DIR).fit(df)

    # Perform Testing (Always)
    X_test = df.drop(columns=[LABEL_COLUMN])
    y_actual = df[LABEL_COLUMN]
    y_pred = predictor.predict(X_test)

    # Evaluation Metrics
    if df[LABEL_COLUMN].dtype in [np.float64, np.int64]:  # Regression Task
        r2 = r2_score(y_actual, y_pred)
        mape = mean_absolute_percentage_error(y_actual, y_pred)
        accuracy = 100 * (1 - mape)  # Interpreted accuracy for regression

        print(f"R² Score: {r2 * 100:.2f}%")
        print(f"Accuracy (based on MAPE): {accuracy:.2f}%")

    else:  # Classification Task
        accuracy = accuracy_score(y_actual, y_pred)
        print(f"Classification Accuracy: {accuracy * 100:.2f}%")



# ** Deployment Function **
def deploy_model(user_input):
    """
    Deploy the trained model, make predictions, and visualize actual vs predicted values.
    """

    # Universal Parameters
    DATA_PATH = filepath()
    target_column = targeted_column(user_input)
    MODEL_DIR = f"./autogluon_model_{target_column}"
    LABEL_COLUMN = target_column
    ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl") 

    if not os.path.exists(MODEL_DIR):
        print("Model not found! Training the model before deploying.")
        build_model_and_test(user_input)
        return

    predictor = TabularPredictor.load(MODEL_DIR)
    print("Model is deployed and ready for inference.")

    df = pd.read_csv(DATA_PATH)

    # Encode categorical labels for consistency
    if df[LABEL_COLUMN].dtype == 'object':
        label_encoder = joblib.load(ENCODER_PATH)  # ✅ Load encoder
        df[LABEL_COLUMN] = label_encoder.transform(df[LABEL_COLUMN])

    X_test = df.drop(columns=[LABEL_COLUMN])
    y_actual = df[LABEL_COLUMN]
    y_pred = predictor.predict(X_test)

    # Determine Task Type (Regression or Classification)
    num_classes = len(np.unique(y_actual))
    
    if num_classes > 10:  # Regression (Assuming >10 unique values is a regression task)
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(y_actual)), y_actual, label="Actual", color="blue", linestyle="-")
        plt.plot(range(len(y_pred)), y_pred, label="Predicted", color="red", linestyle="--")
        plt.xlabel("Sample Index")
        plt.ylabel(LABEL_COLUMN)
        plt.title("Actual vs Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        if num_classes == 2:
            # Bar plot for binary classification
            unique_labels = np.unique(y_actual)
            cm = confusion_matrix(y_actual, y_pred)
            class_labels = [str(label) for label in unique_labels]

            plt.figure(figsize=(6, 4))
            sns.barplot(x=class_labels, y=cm.diagonal(), palette="Blues")
            plt.xlabel("Class")
            plt.ylabel("Correct Predictions")
            plt.title("Correct Predictions per Class")
            plt.show()
        else:
            # Confusion Matrix for multiclass classification
            cm = confusion_matrix(y_actual, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.show()

    print("\nSample Predictions:")
    comparison_df = pd.DataFrame({"Actual": y_actual, "Predicted": y_pred})
    print(comparison_df.head(20))


def predict_custom_input(user_input):
    """
    Collect raw feature inputs from the user, transform them, and make predictions using the trained model.
    """
    # Load dataset to extract feature names
    df = pd.read_csv(filepath())
    print(f"Sample Dataset:\n{df.head()}")
    target_column = targeted_column(user_input)
    feature_names = [col for col in df.columns if col != target_column]

    # Check if model directory exists
    model_dir = f"./autogluon_model_{target_column}"
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} not found!")
        return None

    # Get user input for each feature
    print("\nEnter values for the following features:")
    user_values = {}
    for feature in feature_names:
        value = input(f"{feature}: ")
        try:
            value = float(value) if "." in value else int(value)
        except ValueError:
            pass
        user_values[feature] = value

    # Convert user input to a DataFrame
    input_df = pd.DataFrame([user_values])

    # Load trained model
    model_dir = f"./autogluon_model_{target_column}"
    predictor = TabularPredictor.load(model_dir)

    # Make prediction
    prediction = predictor.predict(input_df)

    print(f"\nPredicted Value: {prediction.values[0]}")
    return prediction.values[0]


if __name__ == "__main__":
    user_input = "Deploy the model"
    build_model_and_test(user_input)