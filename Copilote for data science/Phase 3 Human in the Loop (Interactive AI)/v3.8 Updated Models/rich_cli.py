import sys
import traceback
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

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
from groq import Groq
from Core_Automation_Engine.Data import get_dataset_path
import os
from ML_Models_Engine_autogluon import *
from config.api_manager import get_api_key


client = Groq(api_key=get_api_key())

ML = SupervisedUniversalMachineLearning()
dashboard = Dashboard()
data_analysis_report = DataAnalysisReport()
console = Console()


def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = traceback.format_exc()
            explanation = explain_error(error_message)
            console.print(Panel.fit(explanation, title=f"[red]❌ Error in {func.__name__}()[/red]", border_style="red"))
    return wrapper


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
            model="openai/gpt-oss-20b",
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


@handle_errors
def safe_execute(operation_function, *args):
    return operation_function(*args)


@handle_errors
def execute_multiple_commands(user_input):
    commands = split_multi_commands(user_input)
    for command in commands:
        if command:
            refined = basic_rag(command)
            console.print(f"[bold cyan]\n🔸 Executing command:[/] {refined}")
            operation = NL_processor(refined)
            route_command(operation, refined)


@handle_errors
def route_command(operation, user_input):
    refined = basic_rag(user_input)
    console.print(f"[bold cyan]\n🔸 Executing command:[/] {refined}")
    operation = NL_processor(refined)

    if operation == "visualize":
        console.print("[green]🧩 Visualizing data...[/green]")
        safe_execute(Visualize_Groq_Input, refined)
    elif operation == "display":
        console.print("[green]📊 Displaying data...[/green]")
        safe_execute(Display_Groq_Input, refined)
    elif operation == "modify":
        console.print("[green]✏️ Modifying data...[/green]")
        safe_execute(Modify_Groq_Input, refined)
    elif operation == "undo":
        console.print("[yellow]⏪ Undoing last change...[/yellow]")
        safe_execute(undo_last_change)
    elif operation == "meaningful_response":
        console.print("[green]💬 Generating a response...[/green]")
        safe_execute(genral_response_chatbot, refined)
    elif operation == "analyze_data":
        console.print("[green]📈 Analyzing data...[/green]")
        safe_execute(data_analysis_report.generate_insights, refined)
    elif operation == "generate_report":
        console.print("[green]📝 Generating PDF report...[/green]")
        safe_execute(data_analysis_report.generate_pdf_report, refined)
    elif operation == "create_dashboard":
        console.print("[green]📊 Creating dashboard...[/green]")
        safe_execute(dashboard.create_dashboard, refined)
    elif operation == "build_model":
        console.print("[green]🤖 Building ML model...[/green]")
        ML.build_model_and_test(user_input), refined(user_input)
    elif operation == "test_model":
        console.print("[green]🧪 Testing ML model...[/green]")
        ML.build_model_and_test(user_input), refined(user_input)
    elif operation == "deploy_model":
        console.print("[green]🚀 Deploying ML model...[/green]")
        ML.deploy_model(user_input), refined(user_input)
    elif operation == "predict_custom_input":
        console.print("[green]🔍 Predicting custom input...[/green]")
        safe_execute(ML.predict_custom_input, refined)
    else:
        console.print("[bold red]⚠️ Unable to determine the operation. Please try again.[/bold red]")



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
