import os
import sys
from NL_processor import NL_processor, split_multi_commands
from Engine_Display_data import Groq_Input as Display_Groq_Input
from Engine_Visualize_data import Groq_Input as Visualize_Groq_Input
from Engine_Modify_data import Groq_Input as Modify_Groq_Input
from ML_Models_Builder import build_model
from ML_Models_Deployer import deploy_model
from ML_Model_Tester import test_model


def execute_multiple_commands(user_input):
    """
    Execute multiple commands sequentially.
    """
    commands = split_multi_commands(user_input)  # Split the input into commands
    for command in commands:
        if command:
            print(f"\nExecuting command: {command}")
            operation = NL_processor(command)
            if operation == "visualize":
                print("Visualizing data...")
                Visualize_Groq_Input(command)
            elif operation == "display":
                print("Displaying data...")
                Display_Groq_Input(command)
            elif operation == "modify":
                print("Modifying data...")
                Modify_Groq_Input(command)
            else:
                print("Unable to determine the operation. Please try again.")

def main():
    while True:
        user_input = input("Enter your request (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Check if the input is for ML model building, deployment, or testing
        if "build" in user_input.lower():
            print("Building ML model using AutoML...")
            build_model(user_input)
        elif "predict" in user_input.lower() or "deploy" in user_input.lower():
            print("Deploying ML model...")
            deploy_model(user_input)
        elif "test" in user_input.lower():
            print("Testing ML model...")
            test_model(user_input)
        else:
            # Existing logic for display, visualize, modify
            operation = NL_processor(user_input)
            if operation == "visualize":
                print("Visualizing data...")
                Visualize_Groq_Input(user_input)
            elif operation == "display":
                print("Displaying data...")
                Display_Groq_Input(user_input)
            elif operation == "modify":
                print("Modifying data...")
                Modify_Groq_Input(user_input)
            else:
                print("Unable to determine the operation. Please try again.")

if __name__ == "__main__":
    main()