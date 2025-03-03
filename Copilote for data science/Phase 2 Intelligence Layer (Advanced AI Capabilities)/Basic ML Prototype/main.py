from NL_processor import NL_processor, split_multi_commands
from Engine_Display_data import Groq_Input as Display_Groq_Input
from Engine_Visualize_data import Groq_Input as Visualize_Groq_Input
from Engine_Modify_data import Groq_Input as Modify_Groq_Input
from NL_processor import genral_response_chatbot
from Engine_Data_analysis import *
from OS_Operations import os_operations
from Data import filepath
import pandas as pd
import time

dashboard = Dashboard()
data_analysis_report = DataAnalysisReport()


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
            elif operation == "meaningful_response":
                print("Generating a response...")
                genral_response_chatbot(user_input)
            elif operation == "analyze_data":
                print("Analyzing data...")
                data_analysis_report.generate_insights(user_input)
            elif operation == "generate_report":
                print("Generating report...")
                data_analysis_report.generate_pdf_report(user_input)
            elif operation == "create_dashboard":
                print("Creating dashboard...")
                dashboard = Dashboard()
                dashboard.create_dashboard(user_input)
            elif operation == "os_operations":
                print("Performing OS operations...")
                os_operations(user_input)

            else:
                print("Unable to determine the operation. Please try again.")

def main():
    while True:
        user_input = input("Enter your request (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Check if the input contains multiple commands
        if "then" in user_input or "and" in user_input or "after" in user_input:
            execute_multiple_commands(user_input)
        else:
            # Single command execution
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
            elif operation == "meaningful_response":
                print("Generating a response...")
                genral_response_chatbot(user_input)
            elif operation == "analyze_data":
                print("Analyzing data...")
                data_analysis_report.generate_insights(user_input)
            elif operation == "generate_report":
                print("Generating PDF report...")
                data_analysis_report.generate_pdf_report(user_input)
            elif operation == "create_dashboard":
                print("Creating dashboard...")
                dashboard.create_dashboard(user_input)
            elif operation == "os_operations":
                print("Performing OS operations...")
                os_operations(user_input)
            else:
                print("Unable to determine the operation. Please try again.")

if __name__ == "__main__":
    main()