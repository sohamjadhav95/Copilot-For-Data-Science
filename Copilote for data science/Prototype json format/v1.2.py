import pandas as pd
from groq import Groq
import json

class ExcelQueryEngine:
    def __init__(self, file_path, api_key):
        self.df = pd.read_excel(file_path)
        self.client = Groq(api_key=api_key)
    
    def process_nlp_query(self, query):
        completion = self.client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "system", "content": "Convert user query into structured JSON format."},
                      {"role": "user", "content": query}],
            temperature=0.3,
            top_p=0.95
        )
        llm_response = json.loads(completion.choices[0].message.content)

        structured_query = {
            "query": {
                "metric": "Units Sold",  # Adjust based on your dataset
                "aggregation": llm_response.get("aggregation", "total"),
                "group_by": llm_response.get("group_by", "Country")
            }
        }
        return structured_query
    
    def execute_query(self, structured_query):
        metric = structured_query.get("query", {}).get("metric")
        aggregation = structured_query.get("query", {}).get("aggregation")
        group_by = structured_query.get("query", {}).get("group_by")
        
        if not metric or not aggregation or not group_by:
            return "Invalid query format"
        
        if aggregation == "total":
            result = self.df.groupby(group_by)[metric].sum()
        elif aggregation == "average":
            result = self.df.groupby(group_by)[metric].mean()
        elif aggregation == "count":
            result = self.df.groupby(group_by)[metric].count()
        else:
            return "Unsupported aggregation type"
        
        return result.to_dict()

# Set file path and API key
file_path = r"C:\\Users\\soham\\Downloads\\Financial Sample.xlsx"
api_key = "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"  # Replace with your actual API key

# Create Engine
engine = ExcelQueryEngine(file_path, api_key)

# User input
user_query = "Show Units Sold per Country"
structured_query = engine.process_nlp_query(user_query)
print("Generated Query:", structured_query)

# Execute and print results
results = engine.execute_query(structured_query)
print("Query Results:", results)
