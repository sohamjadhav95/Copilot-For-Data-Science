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
            messages=[
                {"role": "system", "content": "Convert user query into structured JSON format. Ensure the output includes 'query' with 'metric', 'aggregation', 'group_by', and optional 'filters'."},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            top_p=0.95
        )

        response_text = completion.choices[0].message.content.strip()

        if not response_text:
            print("❌ NLP Model returned an empty response!")
            return {"query": {}}

        try:
            structured_query = json.loads(response_text)
            
            # Ensure correct structure
            if "query" not in structured_query:
                structured_query = {
                    "query": {
                        "metric": structured_query.get("request", "Sales"),
                        "aggregation": "average",
                        "group_by": structured_query.get("group_by", "Category"),
                        "filters": structured_query.get("filters", {})
                    }
                }
            
            return structured_query
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing NLP response: {e}")
            print(f"🔍 Raw response: {response_text}")  
            return {"query": {}}  

    
    def execute_query(self, structured_query):
        query_data = structured_query.get("query", {})
        metric = query_data.get("metric")
        aggregation = query_data.get("aggregation")
        group_by = query_data.get("group_by")
        filters = query_data.get("filters", {})
        
        if not metric or not aggregation or not group_by:
            return "Invalid query format"
        
        # Apply filters if present
        df_filtered = self.df
        for column, value in filters.items():
            df_filtered = df_filtered[df_filtered[column] == value]
        
        if aggregation == "total":
            result = df_filtered.groupby(group_by)[metric].sum()
        elif aggregation == "average":
            result = df_filtered.groupby(group_by)[metric].mean()
        elif aggregation == "count":
            result = df_filtered.groupby(group_by)[metric].count()
        elif aggregation == "min":
            result = df_filtered.groupby(group_by)[metric].min()
        elif aggregation == "max":
            result = df_filtered.groupby(group_by)[metric].max()
        elif aggregation == "std":
            result = df_filtered.groupby(group_by)[metric].std()
        elif aggregation == "median":
            result = df_filtered.groupby(group_by)[metric].median()
        else:
            return "Unsupported aggregation type"
        
        return result.to_dict()

# Set file path and API key
file_path = r"C:\\Users\\soham\\Downloads\\Financial Sample.xlsx"
api_key = "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"  # Replace with your actual API key

# Create Engine
engine = ExcelQueryEngine(file_path, api_key)

# User input
user_query = "Show the average Units Sold per Product where Country is USA"
structured_query = engine.process_nlp_query(user_query)
print("Generated Query:", structured_query)

# Execute and print results
results = engine.execute_query(structured_query)
print("Query Results:", results)
