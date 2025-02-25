import pandas as pd
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
from Data import Data_rows, filepath
from NL_processor import result_response

client = Groq(api_key="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")

def build_model(user_input):
    """
    Build a machine learning model using Scikit-learn.
    """
    try:
        # Load the dataset
        data_path = filepath()
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully. Columns:", df.columns.tolist())

        # Extract target variable from user input
        prompt = (
            f"Extract the target variable from the following input: {user_input}\n"
            f"Refer to the dataset columns for context: {df.columns.tolist()}\n"
            "Respond ONLY with the target variable name."
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
        target = completion.choices[0].message.content.strip()
        print("Target variable extracted:", target)

        # Check if the target variable exists
        if target not in df.columns:
            print(f"Error: Target variable '{target}' not found in the dataset.")
            return

        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Determine if the task is classification or regression
        if df[target].dtype == "object":  # Classification
            print("Building classification model...")
            model = RandomForestClassifier()
        else:  # Regression
            print("Building regression model...")
            model = RandomForestRegressor()

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        if df[target].dtype == "object":  # Classification
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy}")
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            print(f"Model MSE: {mse}")

        # Save the model
        model_path = f"{target}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Generate response
        result_response(user_input, f"Model built and saved to {model_path}")

    except Exception as e:
        print(f"Error in build_model: {str(e)}")