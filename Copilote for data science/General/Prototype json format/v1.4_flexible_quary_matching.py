import pandas as pd
from groq import Groq
import json
from fuzzywuzzy import process  # For fuzzy matching

class ExcelQueryEngine:
    def __init__(self, file_path, api_key):
        self.df = pd.read_excel(file_path)
        self.client = Groq(api_key=api_key)
        self.column_map = {col.lower(): col for col in self.df.columns}  # Lowercase mapping
    
    def fuzzy_match_column(self, query_col):
        """Finds the closest matching column name in the dataset."""
        query_col = query_col.lower()
        best_match, score = process.extractOne(query_col, self.column_map.keys())
        return self.column_map.get(best_match, query_col) if score > 70 else None  # Threshold 70%
    
    def process_nlp_query(self, query):
        """Processes the NLP-generated query into a structured format."""
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
                "metric": self.fuzzy_match_column(llm_response.get("metric", "Units Sold")),
                "aggregation": llm_response.get("aggregation", "total"),
                "group_by": self.fuzzy_match_column(llm_response.get("group_by", "Country"))
            }
        }
        return structured_query
    
    def execute_query(self, structured_query):
        """Executes a structured query on the Excel dataset."""
        query_data = structured_query.get("query", {})
        
        metric = query_data.get("metric")
        aggregation = query_data.get("aggregation")
        group_by = query_data.get("group_by")
        
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
user_query = "Find the highest Sales per Product"
structured_query = engine.process_nlp_query(user_query)
print("Generated Query:", structured_query)

# Execute and print results
results = engine.execute_query(structured_query)
print("Query Results:", results)
