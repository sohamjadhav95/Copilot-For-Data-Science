import os
from groq import Groq
import pandas as pd

class ExcelQueryEngine:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)

        # 🔹 Hardcode your API key here (Replace with your actual API key)
        api_key = "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"

        self.client = Groq(api_key=api_key)

    def process_nlp_query(self, user_query):
        """Uses Qwen-2.5-32B to convert natural language into structured queries."""
        completion = self.client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[
                {"role": "system", "content": "You are an AI that converts natural language queries into structured JSON queries."},
                {"role": "user", "content": f"Convert this query into JSON format: {user_query}"}
            ],
            temperature=0.2,  # Lower temperature for more deterministic output
            max_tokens=512,
            top_p=0.95,
            stream=False,
            stop=None,
        )
        structured_query = completion.choices[0].message.content.strip()
        return structured_query

if __name__ == "__main__":
    file_path = r"C:\\Users\\soham\\Downloads\\Financial Sample.xlsx"
    df_engine = ExcelQueryEngine(file_path)

    user_query = "Show total Units Sold in India"
    structured_query = df_engine.process_nlp_query(user_query)
    print("Generated Query:", structured_query)
