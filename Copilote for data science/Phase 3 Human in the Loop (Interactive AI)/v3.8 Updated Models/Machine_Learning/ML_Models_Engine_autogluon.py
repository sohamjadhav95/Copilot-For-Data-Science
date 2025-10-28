import pandas as pd
import os
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
from pycaret.clustering import setup, create_model, assign_model, save_model, load_model
from sklearn.preprocessing import StandardScaler
import joblib
from tpot import TPOTClassifier
from sklearn.cluster import KMeans
from kneed import KneeLocator  # Import for automatic knee detection
from pycaret.clustering import *
from sklearn.preprocessing import LabelEncoder
from config.api_manager import get_api_key
import sys
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.8 Updated Models\Core_Automation_Engine")
from Data import filepath


client = Groq(api_key=get_api_key())

def task_type(user_input):
    """
    Determines the machine learning task type based on the dataset and user input.
    Returns: 'Regression', 'Binary Classification', 'Multiclass Classification', or 'Clustering'
    """
    df = pd.read_csv(filepath())
    dtype = df.dtypes

    # Create prompt for Groq API
    prompt = (
        f"Determine the machine learning task type based on this information:\n"
        f"User Input: {user_input}\n"
        f"Data types: {dtype}\n"
        f"Sample dataset: {df.head(100)}\n\n"
        f"Choose ONLY ONE from these options:\n"
        f"- Regression: if target is continuous numeric\n"
        f"- Binary Classification: if target has exactly 2 unique values\n"
        f"- Multiclass Classification: if target has more than 2 unique categorical values\n"
        f"- Clustering: if no clear target column exists\n"
        f"Respond ONLY with the chosen task type."
    )
    

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=50,
        top_p=0.9,
        stream=False,
        stop=None,
    )
    task = completion.choices[0].message.content.strip()
    print(f"Task type Determined From Dataset: {task}")
    
    # Validate response
    # valid_tasks = ['Regression', 'Binary Classification', 'Multiclass Classification', 'Clustering']
    
    if "Regression" in task:
        return "Regression"
    elif "Binary Classification" in task:
        return "Binary Classification"
    elif "Multiclass Classification" in task:
        return "Multiclass Classification"
    elif "Clustering" in task:
        return "Clustering"
    else:
        raise ValueError(f"Invalid task type received: {task}")

class SupervisedUniversalMachineLearning:
    def targeted_column(self, user_input):
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
                model="meta-llama/llama-4-scout-17b-16e-instruct",
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
            
    #======================================================================================

    def build_model_and_test(self, user_input, time_limit=720):
        """
        Trains an ML model using AutoGluon if it does not exist and always performs testing.
        """
        if task_type(user_input) == "Clustering":
            USL = UnsupervisedMachineLearning()
            USL.build_model_and_test()
            return
        else:
            # Universal Parameters
            DATA_PATH = filepath()
            target_column = self.targeted_column(user_input)
            MODEL_DIR = f"./autogluon_model_{target_column}"
            LABEL_COLUMN = target_column
            ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

            # Load dataset
            task = task_type(user_input)
            print(f"Task Type for Supervised: {task}")
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
                print(f"Training new model (time limit: 12 minutes) and saving to {MODEL_DIR}.")
                os.makedirs(MODEL_DIR, exist_ok=True)

                # Encode categorical target variable if needed
                if df[LABEL_COLUMN].dtype == 'object':
                    label_encoder = LabelEncoder()
                    df[LABEL_COLUMN] = label_encoder.fit_transform(df[LABEL_COLUMN])
                    joblib.dump(label_encoder, ENCODER_PATH)  # ✅ Save encoder

                # Train Model with a 12-minute time limit
                predictor = TabularPredictor(label=LABEL_COLUMN, path=MODEL_DIR).fit(df, time_limit=720)

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
    def deploy_model(self, user_input):
        """
        Deploy the trained model, make predictions, and visualize actual vs predicted values.
        """
        if task_type(user_input) == "Clustering":
            USL = UnsupervisedMachineLearning()
            USL.deploy_model()
            return
        else:
            # Universal Parameters
            DATA_PATH = filepath()
            target_column = self.targeted_column(user_input)
            MODEL_DIR = f"./autogluon_model_{target_column}"
            LABEL_COLUMN = target_column
            ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl") 

            if not os.path.exists(MODEL_DIR):
                print("Model not found! Training the model before deploying.")
                self.build_model_and_test(user_input)
                self.deploy_model(user_input)
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
                    # Grouped Bar Plot: Actual vs Predicted Count
                    actual_counts = pd.Series(y_actual).value_counts().sort_index()
                    predicted_counts = pd.Series(y_pred).value_counts().sort_index()

                    comparison_df = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts}).fillna(0)
                    comparison_df.plot(kind="bar", figsize=(6, 4), colormap="coolwarm")
                    plt.xlabel("Class")
                    plt.ylabel("Count")
                    plt.title("Actual vs Predicted Class Distribution")
                    plt.xticks(rotation=0)
                    plt.legend()
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


    def predict_custom_input(self, user_input):
        """
        Collect raw feature inputs from the user, transform them, and make predictions using the trained model.
        """
        if task_type(user_input) == "Clustering":
            USL = UnsupervisedMachineLearning()
            USL.predict_custom_input()
            return
        else:
            # Load dataset to extract feature names
            df = pd.read_csv(filepath())
            print(f"Sample Dataset:\n{df.head()}")
            target_column = self.targeted_column(user_input)
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

#======================================================================================


class UnsupervisedMachineLearning:
    def __init__(self, model_name="clustering_model"):
        self.data_path = filepath()
        self.model_name = model_name
        self.scaler_path = "scaler.pkl"
        self.clustered_data_path = "clustered_data.csv"

    def find_optimal_clusters(self, max_clusters=10):
        """
        Finds the optimal number of clusters using the Elbow Method.

        Parameters:
        - max_clusters: Maximum number of clusters to test (default=10).

        Returns:
        - optimal_k: The optimal number of clusters detected.
        """
        # Load dataset
        df = pd.read_csv(self.data_path)

        # Standardize features
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # Compute WCSS (Within-Cluster Sum of Squares) for different cluster values
        wcss = []
        cluster_range = range(1, max_clusters + 1)

        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_scaled)
            wcss.append(kmeans.inertia_)

        # Find the optimal number of clusters using KneeLocator
        knee_locator = KneeLocator(cluster_range, wcss, curve="convex", direction="decreasing")
        optimal_k = knee_locator.elbow  # Extract the elbow point

        # Plot the Elbow Graph
        plt.figure(figsize=(8, 6))
        plt.plot(cluster_range, wcss, marker="o", linestyle="-", color="b", label="WCSS")
        plt.axvline(optimal_k, linestyle="--", color="r", label=f"Optimal K = {optimal_k}")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
        plt.title("Elbow Method for Optimal Clusters")
        plt.xticks(cluster_range)
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Optimal number of clusters detected: {optimal_k}")
        return optimal_k

    def build_model_and_test(self):
        """
        Automatically detects the optimal number of clusters, trains a model, and saves clustered data.
        """
        # Get optimal number of clusters
        optimal_k = self.find_optimal_clusters()
        
        # Load dataset
        df = pd.read_csv(self.data_path)

        # Standardize features
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # Train KMeans model with detected optimal clusters
        model = KMeans(n_clusters=optimal_k, random_state=42)
        df["Cluster"] = model.fit_predict(df_scaled)

        # Save model and scaler
        joblib.dump(model, self.model_name + ".pkl")
        joblib.dump(scaler, self.scaler_path)

        # Save clustered data
        df.to_csv(self.clustered_data_path, index=False)

        print(f"Model trained with {optimal_k} clusters and saved as '{self.model_name}.pkl'. Clustered data saved as '{self.clustered_data_path}'.")

        # If model exists, test accuracy
        if os.path.exists(self.model_name + ".pkl"):
            predicted_clusters = model.predict(df_scaled)
            true_clusters = df["Cluster"].values

            accuracy = accuracy_score(true_clusters, predicted_clusters) * 100
            print(f"Model Accuracy: {accuracy:.2f}%")

    def deploy_model(self):
        """
        Deploys the trained clustering model and visualizes the clusters.
        """
        if not os.path.exists(self.model_name + ".pkl") or not os.path.exists(self.clustered_data_path):
            print("Model or clustered data not found! Training the model first.")
            self.build_model_and_test()
            self.deploy_model()
            return

        # Load data
        df = pd.read_csv(self.clustered_data_path)

        # Scatter plot of clusters
        plt.figure(figsize=(8, 6))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df["Cluster"], cmap="viridis", alpha=0.6)
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.title("Cluster Visualization")
        plt.colorbar(label="Cluster")
        plt.show()

        print("Model deployed successfully!")

    def predict_custom_input(self):
        """
        Takes user input interactively, scales it, and predicts its cluster using the trained model.
        """
        if not os.path.exists(self.model_name + ".pkl") or not os.path.exists(self.scaler_path):
            print("Error: Model or scaler not found!")
            return

        # Load model and scaler
        model = joblib.load(self.model_name + ".pkl")
        scaler = joblib.load(self.scaler_path)

        # Read dataset to get feature names
        df = pd.read_csv(self.data_path)
        feature_names = df.columns

        # Collect user input one by one
        user_input = []
        print("\nEnter feature values:")
        for feature in feature_names:
            value = float(input(f"{feature}: "))
            user_input.append(value)

        # Convert input into DataFrame
        input_df = pd.DataFrame([user_input], columns=feature_names)

        # Scale input data
        input_scaled = scaler.transform(input_df)

        # Predict cluster
        prediction = model.predict(input_scaled)
        print(f"\nPredicted Cluster: {prediction[0]}")
        return prediction[0]

#======================================================================================
if __name__ == "__main__":  
    def main():
        user_input = "Train Machine learning Model"
        if task_type(user_input) == "Clustering":
            USL = UnsupervisedMachineLearning()
            USL.build_model_and_test()
            USL.deploy_model()
            USL.predict_custom_input()
        else:
            ML = SupervisedUniversalMachineLearning()
            ML.build_model_and_test(user_input)
            ML.deploy_model(user_input)
            ML.predict_custom_input(user_input)
    main()

