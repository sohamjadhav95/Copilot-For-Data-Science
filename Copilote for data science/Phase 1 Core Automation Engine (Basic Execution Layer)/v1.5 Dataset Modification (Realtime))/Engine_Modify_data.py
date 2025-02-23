import re
from groq import Groq
from Data import Data_rows

# Configure the Groq API with your API key
client = Groq(api_key="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")

def Groq_Input(user_input):
    try:
        first_100_rows, last_100_rows, data = Data_rows()
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
            print("No valid code detected!")
            return

        print("Generated Code:\n", generated_code)

        try:
            print("\nExecuting Code...")
            exec(generated_code)  # This modifies the CSV file
            print("\nData modified and saved successfully!")
        except Exception as e:
            print("\nExecution Error:", e)
    except Exception as e:
        print(f"Error in Groq_Input (Modify): {e}")