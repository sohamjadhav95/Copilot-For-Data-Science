import os
import shutil
import re
from groq import Groq
from Data import Data_rows, filepath
from SQL_Operations import SQLExecutor
from NL_processor import result_response

# Configure the Groq API with your API key
client = Groq(api_key="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")

first_100_rows, last_100_rows = Data_rows()
data = filepath()
backup_path = data + ".backup"  # Backup file path

def create_backup():
    """Creates a backup of the dataset before modification."""
    if os.path.exists(data):
        shutil.copy(data, backup_path)
    print("Backup Is Created Before Modification, Use Undo to restore changes")

def undo_last_change():
    """Restores the dataset from the last backup."""
    if os.path.exists(backup_path):
        shutil.copy(backup_path, data)
        print("Undo successful: Data restored from the last backup.")
    else:
        print("No backup found! Cannot undo.")

def Groq_Input(user_input):
    sql_executor = SQLExecutor()
    first_100_rows, last_100_rows = Data_rows()

    # Create backup before applying modifications
    create_backup()
    
    data = filepath()

    # First try SQL approach
    sql_prompt = (
        f"Refer this dataset: {first_100_rows}, {last_100_rows}\n"
        f"Convert this modification request to SQL: {user_input}. "
        "Use UPDATE, INSERT, or DELETE statements. Respond ONLY with SQL code."
        f"Take this csv file: {data} as input for data in your code."
    )
    completion = client.chat.completions.create(
        model= "qwen-2.5-coder-32b",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0.4,
        max_tokens=1024
    )
    sql_query = completion.choices[0].message.content
    
    if sql_query:
        # Extract SQL from markdown block
        sql_match = re.search(r"```sql\n(.*?)\n```", sql_query, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            code_match = re.search(r"```\n(.*?)\n```", sql_query, re.DOTALL)
            if code_match:
                sql_query = code_match.group(1).strip()
        
        _, success = sql_executor.execute_sql(sql_query)
        if success:
            sql_executor.save_changes()
            print("Modification successful via SQL!")
            result_response(user_input, sql_query)
            return
    
    # Fallback to code generation
    original_code_generation_approach(user_input)


def original_code_generation_approach(user_input):
    try:
        first_100_rows, last_100_rows = Data_rows()
        data = filepath()
        if first_100_rows is None or last_100_rows is None:
            print("Error: Unable to load data.")
            return

        prompt = (
            f"Dataset's first 100 rows: {first_100_rows}\nLast 100 rows: {last_100_rows}\n"
            f"Generate Python code to modify the dataset as per: '{user_input}'\n"
            f"Read the CSV file from this path: '{data}', make changes, and overwrite it.\n"
            f"Ensure the code:\n"
            f"1. Uses pandas to load/save the CSV.\n"
            f"2. Saves changes with df.to_csv('{data}', index=False).\n"
            f"3. Only performs modification tasks.\n"
        )

        completion = client.chat.completions.create(
            model="qwen-2.5-coder-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        generated_code = completion.choices[0].message.content

        # Extract Python code block
        code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1).strip()
        else:
            print("No valid Modification Logic detected!")
            return

        try:
            print("\nExecuting the Modification Operation...\n")
            exec(generated_code)  # This modifies the CSV file
            print("\nData modified and saved successfully!")
            result_response(user_input, generated_code)
        except Exception as e:
            print("\nExecution Error:", e)

            #Fallback to code generation
            generate_code_error_handling(user_input, generated_code, e)
    except Exception as e:
        print(f"Error in Groq_Input (Modify): {e}")


def generate_code_error_handling(user_input, generated_code, e):
    try:
        first_100_rows, last_100_rows = Data_rows()
        data = filepath()
        if first_100_rows is None or last_100_rows is None:
            print("Error: Unable to load data.")
            return

        prompt = (
            f"See the error in this generated code: {generated_code}\n error: {e}\n"
            f"Solve this error and regenerate the code and make sure it works.\n"
            f"Refer the dataset again for reference: {first_100_rows}, {last_100_rows}\n"
            f"Take this csv file: {data} as input for data in your code.\n"
        )

        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        generated_code = completion.choices[0].message.content

        # Find the first occurrence of a Python code block
        code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)

        if code_match:
            generated_code = code_match.group(1).strip()  # Extract the valid Python code
        else:
            print("No valid Modification Logic detected in response!")
            return

        try:
            print("\nExecuting the Modification Operation...\n")
            exec(generated_code)
            print("\nTask completed successfully!")
            result_response(user_input, generated_code)
        except Exception as e:
            print("\nThe Model Is Not Capable To Perform This Operation Currently:")
            print(e)
    except Exception as e:
        print(f"An error occurred in Groq_Input: {e}")

