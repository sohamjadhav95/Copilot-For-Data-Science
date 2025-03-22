import re
from groq import Groq
import pandas as pd
from Data import Data_rows, filepath
from SQL_Operations import SQLExecutor
from NL_processor import result_response

# Configure the Groq API with your API key
client = Groq(api_key="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")  # Replace with your Groq API key

def Groq_Input(user_input):
    sql_executor = SQLExecutor()
    first_100_rows, last_100_rows = Data_rows()
    data = filepath()
    
    # First try SQL approach
    sql_prompt = (
        f"Refer this dataset: {first_100_rows}, {last_100_rows}\n"
        f"Take this csv file: {data} as input for data in your code.\n"
        f"Convert this request to SQL query: {user_input}. Respond ONLY with SQL code."
    )

    completion = client.chat.completions.create(
        model= "qwen-2.5-coder-32b",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0.4,
        max_tokens=1024
    )
    sql_query = completion.choices[0].message.content
    
    if sql_query:
        # Extract SQL code from markdown block
        sql_match = re.search(r"```sql\n(.*?)\n```", sql_query, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            # Check for generic code block
            code_match = re.search(r"```\n(.*?)\n```", sql_query, re.DOTALL)
            if code_match:
                sql_query = code_match.group(1).strip()
        
        result, success = sql_executor.execute_sql(sql_query)
        if success:
            print(result)
            result_response(user_input, result)
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
            f"See this dataset's First and Last 100 rows you have provided: {first_100_rows}, {last_100_rows}\n"
            f"Based on that For Whole Dataset Generate Python code to Display: {user_input}.\n"
            f"Take this csv file: {data} as input for data in your code.\n"
            f"Make sure that only 'Display' operation is complete by referring the dataset.\n"
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

        # Find the first occurrence of a Python code block
        code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)

        if code_match:
            generated_code = code_match.group(1).strip()  # Extract the valid Python code
        else:
            print("No valid Display Logic detected in response!")
            return

        try:
            print("\nExecuting the Display Operation...\n")
            exec(generated_code)
            print("\nTask completed successfully!")
            result_response(user_input, generated_code)
        except Exception as e:
            print("\nAn error occurred while executing the Display Operation:")
            print(e)
            
            #Fallback to code generation
            generate_code_error_handling(user_input, generated_code, e)
    except Exception as e:
        print(f"An error occurred in Groq_Input: {e}")


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
            print("No valid Display Logic detected in response!")
            return

        try:
            print("\nExecuting the Display Operation...\n")
            exec(generated_code)
            print("\nTask completed successfully!")
            result_response(user_input, generated_code)
        except Exception as e:
            print("\nThe Model Is Not Capable To Perform This Operation Currently:")
            print(e)
    except Exception as e:
        print(f"An error occurred in Groq_Input: {e}")