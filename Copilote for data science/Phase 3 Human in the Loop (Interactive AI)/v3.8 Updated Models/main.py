import sys
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.8 Updated Models\Machine_Learning")
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.8 Updated Models\Core_Automation_Engine")
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.8 Updated Models\Retrival_Agumented_Generation")

from NL_processor import NL_processor, split_multi_commands
from Engine_Display_data import Groq_Input as Display_Groq_Input
from Engine_Visualize_data import Groq_Input as Visualize_Groq_Input
from Engine_Modify_data import Groq_Input as Modify_Groq_Input, undo_last_change
from NL_processor import genral_response_chatbot
from Engine_Data_analysis import *
from rag_command_parser import basic_rag
from Core_Automation_Engine.Data import get_dataset_path
import os
import pandas as pd
import time
import traceback
from openai import OpenAI
from ML_Models_Engine_autogluon import *
from config.api_manager import get_api_key


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)
ML = SupervisedUniversalMachineLearning()


# Universal Error Handler
def handle_errors(func):
    """Decorator to handle errors for all functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = traceback.format_exc()  # Capture full error traceback
            explanation = explain_error(error_message)  # Get dynamic explanation
            print(f"\n[ERROR] An error occurred in {func.__name__}:\n{explanation}")
    return wrapper

# Use Groq API to dynamically explain errors
def explain_error(error_message):
    try:
        prompt = f"""
        You are an AI assistant that helps developers debug Python errors.
        Given the following error message, provide an explanation and a solution.

        Error Message:
        {error_message}

        Explanation and Suggested Fix:
        """

        completion = client.chat.completions.create(
            model="qwen/qwen3-coder-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        return completion.choices[0].message.content.strip()
    except Exception:
        return "Error explanation service is unavailable. Please check your API connection."

# Safe Execution Wrapper
@handle_errors
def safe_execute(operation_function, *args):
    """Safely executes any function with error handling."""
    return operation_function(*args)

# Initialize required objects
dashboard = Dashboard()
data_analysis_report = DataAnalysisReport()

@handle_errors
def execute_multiple_commands(user_input):
    """Execute multiple commands sequentially with error handling."""
    commands = split_multi_commands(user_input)  # Split the input into commands
    for command in commands:
        if command:
            refined_user_input_by_rag = basic_rag(command)
            print(f"\nExecuting command: {refined_user_input_by_rag}")
            operation = NL_processor(refined_user_input_by_rag)
            route_command(operation, refined_user_input_by_rag)

@handle_errors
def route_command(operation, user_input):
    """Routes command execution based on operation type."""

    refined_user_input_by_rag = basic_rag(user_input)
    print(f"\nExecuting command: {refined_user_input_by_rag}")
    operation = NL_processor(refined_user_input_by_rag)

    if operation == "visualize":
        print("Visualizing data...")
        safe_execute(Visualize_Groq_Input, refined_user_input_by_rag)
    elif operation == "display":
        print("Displaying data...")
        safe_execute(Display_Groq_Input, refined_user_input_by_rag)
    elif operation == "modify":
        print("Modifying data...")
        safe_execute(Modify_Groq_Input, refined_user_input_by_rag)
    elif operation == "undo":
        print("Undoing last modification...")
        safe_execute(undo_last_change)
    elif operation == "meaningful_response":
        print("Generating a response...")
        safe_execute(genral_response_chatbot, refined_user_input_by_rag)
    elif operation == "analyze_data":
        print("Analyzing data...")
        safe_execute(data_analysis_report.generate_insights, refined_user_input_by_rag)
    elif operation == "generate_report":
        print("Generating PDF report...")
        safe_execute(data_analysis_report.generate_pdf_report, refined_user_input_by_rag)
    elif operation == "create_dashboard":
        print("Creating dashboard...")
        safe_execute(dashboard.create_dashboard, refined_user_input_by_rag)
    elif operation == "build_model":
        print("Building ML model...")
        ML.build_model_and_test(user_input), refined_user_input_by_rag(user_input)
    elif operation == "test_model":
        print("Testing ML model...")
        ML.build_model_and_test(user_input), refined_user_input_by_rag(user_input)
    elif operation == "deploy_model":
        print("Deploying ML model...")
        ML.deploy_model(user_input), refined_user_input_by_rag(user_input)
    elif operation == "predict_custom_input":
        print("Predicting custom input...")
        safe_execute(ML.predict_custom_input, refined_user_input_by_rag)
    else:
        print("Unable to determine the operation. Please try again.")


@handle_errors
def main():
    """Main loop to receive user input and execute commands."""

    # Reset dataset path each time the program runs (optional)
    dataset_config_path = os.path.join("config", "dataset_path.txt")
    if os.path.exists(dataset_config_path):
        os.remove(dataset_config_path)

    # Prompt user to enter dataset path first
    print("🔍 Let's begin by selecting your dataset.")
    dataset_path = get_dataset_path()
    print(f"📁 Using dataset: {dataset_path}")

    # Main command loop
    while True:
        user_input = input("\n🧠 Enter your request (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("👋 Exiting Copilot. Goodbye!")
            break

        if "then" in user_input or "and" in user_input or "after" in user_input:
            execute_multiple_commands(user_input)
        else:
            operation = NL_processor(user_input)
            route_command(operation, user_input)


if __name__ == "__main__":
    main()