from autogluon.tabular import TabularPredictor
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_percentage_error, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Universal Parameters
DATA_PATH = r"C:\Users\soham\Downloads\iris_synthetic_data.csv"
MODEL_DIR = "./autogluon_model_iris_label"
LABEL_COLUMN = "label"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Encode categorical target variable if needed
if df[LABEL_COLUMN].dtype == 'object':
    label_encoder = LabelEncoder()
    df[LABEL_COLUMN] = label_encoder.fit_transform(df[LABEL_COLUMN])

# Train Model
predictor = TabularPredictor(label=LABEL_COLUMN, path=MODEL_DIR).fit(df)

# Make Predictions
preds = predictor.predict(df.drop(columns=[LABEL_COLUMN]))

# Evaluation Metrics
if df[LABEL_COLUMN].dtype in [np.float64, np.int64]:  # Regression Task
    r2 = r2_score(df[LABEL_COLUMN], preds)
    mape = mean_absolute_percentage_error(df[LABEL_COLUMN], preds)
    accuracy = 100 * (1 - mape)  # Interpreted accuracy for regression

    print(f"R² Score: {r2 * 100:.2f}%")
    print(f"Accuracy (based on MAPE): {accuracy:.2f}%")

else:  # Classification Task
    accuracy = accuracy_score(df[LABEL_COLUMN], preds)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")


# ** Deployment Function **
def deploy_model():
    """
    Deploy the trained model, make predictions, and visualize actual vs predicted values.
    """
    if not os.path.exists(MODEL_DIR):
        print("Model not found! Train the model before deploying.")
        return

    predictor = TabularPredictor.load(MODEL_DIR)
    print("Model is deployed and ready for inference.")

    df = pd.read_csv(DATA_PATH)

    # Encode categorical labels for consistency
    if df[LABEL_COLUMN].dtype == 'object':
        df[LABEL_COLUMN] = label_encoder.transform(df[LABEL_COLUMN])

    X_test = df.drop(columns=[LABEL_COLUMN])
    y_actual = df[LABEL_COLUMN]
    y_pred = predictor.predict(X_test)

    # Visualization for Regression
    if df[LABEL_COLUMN].dtype in [np.float64, np.int64]:
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
        # Confusion Matrix for Classification
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


# Call deploy function
deploy_model()
