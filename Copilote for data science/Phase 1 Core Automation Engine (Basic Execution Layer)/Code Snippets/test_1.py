import re
from groq import Groq
import pandas as pd
from Data import Data_rows
from SQL_Operations import SQLExecutor

# Configure the Groq API with your API key
client = Groq(api_key="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")  # Replace with your Groq API key

def Groq_Input(user_input):
    sql_executor = SQLExecutor()
    first_100_rows, last_100_rows, data = Data_rows()
    
    # First try SQL approach
    sql_prompt = (
        f"Refer this dataset: {first_100_rows}, {last_100_rows}\n"
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
            return