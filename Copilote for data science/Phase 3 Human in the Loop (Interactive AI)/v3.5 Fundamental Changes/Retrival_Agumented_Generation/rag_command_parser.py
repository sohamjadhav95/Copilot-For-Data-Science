import csv
from groq import Groq
import pandas as pd
import os

client = Groq(api_key="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")

CSV_FILE = r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.5 Fundamental Changes\Retrival_Agumented_Generation\commands_database.csv"

# Initialize CSV file if it doesn't exist
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Command'])  # Add header

def store_command(command):
    """Append the command to the CSV file."""
    try:
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([command])
    except Exception as e:
        print(f"Error storing command: {str(e)}")

def get_last_commands(n=5):
    """Get the last n commands from the CSV file."""
    try:
        if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
            # Read CSV with explicit column name
            df = pd.read_csv(CSV_FILE, names=['Command'], skiprows=1)
            # Filter out 'No Command' entries and empty rows
            df = df[~df['Command'].isin(["'No Command'", "", "Command"])]
            # Get last n commands
            last_commands = df.tail(n)
            return last_commands['Command'].tolist()
        return []
    except Exception as e:
        print(f"Error reading commands: {str(e)}")
        return []

def modify_command_api(user_input):
    """Use the API to modify the command based on recent commands."""
    last_commands = get_last_commands(5)
    context = "\n".join(last_commands) if last_commands else "No previous commands"

    prompt = f"""
    You are an AI assistant that refines user commands based on previous instructions Like Windows Memory Buffer.
    Here are the last few commands given by the user:
    {context}

    The user has now given a new command: "{user_input}"

    Rules for command modification:
    Importent : If there is no need to modify command return as it is.
    1. If the command is a "restore" command and follows a "delete" command, include the specific details from the delete command
    2. Maintain context between related commands (delete/restore, copy/paste, etc.)
    3. If no previous commands are referenced, return the original command as is

    Return only the modified command, nothing else.
    """

    try:
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=100,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        response = completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        return f"Error: {str(e)}"

def basic_rag(user_input):
    """Process user input, modify it using API, and store the result."""
    try:
        modified_command = modify_command_api(user_input)
        if not modified_command.startswith("Error:"):
            store_command(modified_command)
            print(f"Modified command: {modified_command}")
        else:
            print(f"Error occurred: {modified_command}")
        return modified_command
    except Exception as e:
        error_msg = f"Error in basic_rag: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    user_input = "Visualize the dataset"
    modified_command = basic_rag(user_input)