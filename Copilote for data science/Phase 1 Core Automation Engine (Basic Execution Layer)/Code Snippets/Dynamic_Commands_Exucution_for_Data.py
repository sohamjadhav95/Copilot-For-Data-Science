import re  # Import for cleaning text
from groq import Groq
import pandas as pd

# Configure the Groq API with your API key
client = Groq(api_key="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")  # Replace with your Groq API key

dataset = pd.read_csv(r"C:\Users\soham\Downloads\synthetic_sales_data.csv")

def Groq_Input(user_input):
    prompt = (
        f"See this dataset you have provided: {dataset}"
        f"Generate Python code to {user_input}.\n"
        f"Make sure that operation is complete by referring the dataset.\n")


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
        print("No valid code detected in response!")
        exit()

    print("Generated Code:\n")
    print(generated_code)

    try:
        print("\nExecuting the Generated Code...\n")
        exec(generated_code)
        print("\nTask completed successfully!")
    except Exception as e:
        print("\nAn error occurred while executing the generated code:")
        print(e)


Groq_Input("Add column as net profit as units sold * sales price")
