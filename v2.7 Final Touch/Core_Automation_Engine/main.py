from NL_processor import NL_processor, split_multi_commands
from Engine_Display_data import Groq_Input as Display_Groq_Input
from Engine_Visualize_data import Groq_Input as Visualize_Groq_Input
from Engine_Modify_data import Groq_Input as Modify_Groq_Input, undo_last_change
from NL_processor import genral_response_chatbot
from Engine_Data_analysis import *
from Data import filepath
import pandas as pd
import time
import traceback
from groq import Groq

import sys
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 2 Intelligence Layer (Advanced AI Capabilities)\v2.7 Final Touch\Machine_Learning")
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 2 Intelligence Layer (Advanced AI Capabilities)\v2.7 Final Touch\System_OS_Operations")
from ML_Models_Engine import deploy_model, build_model, test_model
from Main import system_operation

client = Groq(api_key="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")

dashboard = Dashboard()
data_analysis_report = DataAnalysisReport()


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
            model="gemma2-9b-it",
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
            print(f"\nExecuting command: {command}")
            operation = NL_processor(command)
            route_command(operation, command)

@handle_errors
def route_command(operation, user_input):
    """Routes command execution based on operation type."""
    if operation == "visualize":
        print("Visualizing data...")
        safe_execute(Visualize_Groq_Input, user_input)
    elif operation == "display":
        print("Displaying data...")
        safe_execute(Display_Groq_Input, user_input)
    elif operation == "modify":
        print("Modifying data...")
        safe_execute(Modify_Groq_Input, user_input)
    elif operation == "undo":
        print("Undoing last modification...")
        safe_execute(undo_last_change)
    elif operation == "meaningful_response":
        print("Generating a response...")
        safe_execute(genral_response_chatbot, user_input)
    elif operation == "analyze_data":
        print("Analyzing data...")
        safe_execute(data_analysis_report.generate_insights, user_input)
    elif operation == "generate_report":
        print("Generating PDF report...")
        safe_execute(data_analysis_report.generate_pdf_report, user_input)
    elif operation == "create_dashboard":
        print("Creating dashboard...")
        safe_execute(dashboard.create_dashboard, user_input)
    elif operation == "build_model":
        print("Building ML model...")
        safe_execute(build_model, user_input)
    elif operation == "test_model":
        print("Testing ML model...")
        safe_execute(test_model, user_input)
    elif operation == "deploy_model":
        print("Deploying ML model...")
        safe_execute(deploy_model, user_input)
    elif operation == "os_operations":
        print("Performing OS operations...")
        safe_execute(system_operation, user_input)
    else:
        print("Unable to determine the operation. Please try again.")

@handle_errors
def main():
    """Main loop to receive user input and execute commands."""
    while True:
        user_input = input("Enter your request (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        if "then" in user_input or "and" in user_input or "after" in user_input:
            execute_multiple_commands(user_input)
        else:
            operation = NL_processor(user_input)
            route_command(operation, user_input)

if __name__ == "__main__":
    main()