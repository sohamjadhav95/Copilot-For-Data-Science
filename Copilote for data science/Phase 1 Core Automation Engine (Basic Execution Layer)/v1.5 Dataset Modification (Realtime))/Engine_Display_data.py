import re
from groq import Groq
import pandas as pd
from Data import Data_rows

# Configure the Groq API with your API key
client = Groq(api_key="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")  # Replace with your Groq API key

def Groq_Input(user_input):
    try:
        first_100_rows, last_100_rows, data = Data_rows()
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
            print("No valid code detected in response!")
            return

        print("Generated Code:\n")
        print(generated_code)

        try:
            print("\nExecuting the Generated Code...\n")
            exec(generated_code)
            print("\nTask completed successfully!")
        except Exception as e:
            print("\nAn error occurred while executing the generated code:")
            print(e)
    except Exception as e:
        print(f"An error occurred in Groq_Input: {e}")