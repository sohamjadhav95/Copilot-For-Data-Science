import pandas as pd
from groq import Groq
import json
from fuzzywuzzy import process

class ExcelQueryEngine:
    def __init__(self, file_path, api_key):
        self.df = pd.read_excel(file_path)
        self.client = Groq(api_key=api_key)
        
        # Convert date columns to datetime format
        for col in self.df.columns:
            if "date" in col.lower() or "time" in col.lower() or "year" in col.lower():
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    def process_nlp_query(self, query):
        """Convert natural language query into structured JSON format using NLP."""
        try:
            completion = self.client.chat.completions.create(
                model="qwen-2.5-32b",
                messages=[
                    {"role": "system", "content": "Convert user query into structured JSON format with metric, aggregation, group_by, filters, sort, and limit."},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                top_p=0.95
            )

            if not completion or not completion.choices or not completion.choices[0].message:
                return {"error": "NLP model returned an empty response!"}

            response_text = completion.choices[0].message.content.strip()

            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return {"error": "NLP model response is not valid JSON!", "response": response_text}

        except Exception as e:
            return {"error": f"Exception occurred: {str(e)}"}

    def fuzzy_match_columns(self, input_columns):
        """Fuzzy match user-requested columns to dataset columns."""
        matched_columns = []
        available_columns = self.df.columns.tolist()
        
        for col in input_columns:
            match, score = process.extractOne(col, available_columns)
            if score > 80:
                matched_columns.append(match)
            else:
                print(f"⚠️ Warning: '{col}' not found in dataset (score: {score}).")
                matched_columns.append(None)
        
        return matched_columns

    def execute_query(self, structured_query):
        df_filtered = self.df.copy()
        
        # Extract query components
        metric = structured_query.get("metric", "")
        aggregation = structured_query.get("aggregation", "")
        group_by = structured_query.get("group_by", [])
        filters = structured_query.get("filters", {})

        # Convert single values to lists for consistency
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Fuzzy match all relevant columns
        metric_match = self.fuzzy_match_columns([metric])[0]
        group_by_matches = self.fuzzy_match_columns(group_by)
        filter_matches = {self.fuzzy_match_columns([col])[0]: value for col, value in filters.items()}

        # Validate metric
        if not metric_match:
            return "❌ Metric not found in dataset!"
        
        # Apply filters
        for col, value in filter_matches.items():
            if col in df_filtered.columns and value is not None:
                df_filtered = df_filtered[df_filtered[col] == value]

        # Apply aggregation
        if aggregation == "sum":
            result = df_filtered.groupby(group_by_matches)[metric_match].sum()
        elif aggregation == "average":
            result = df_filtered.groupby(group_by_matches)[metric_match].mean()
        elif aggregation == "count":
            result = df_filtered.groupby(group_by_matches)[metric_match].count()
        else:
            return "❌ Unsupported aggregation type!"

        return result.to_dict()

# Set file path and API key
file_path = r"C:\Users\soham\Downloads\synthetic_sales_data.xlsx"
api_key = "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"

# Create Engine
engine = ExcelQueryEngine(file_path, api_key)

# Example Queries
user_queries = [
    "Show Amarila Units sold in France."
]

# Execute Queries
for user_query in user_queries:
    structured_query = engine.process_nlp_query(user_query)
    print(f"\nGenerated Query: {structured_query}")
    
    results = engine.execute_query(structured_query)
    print(f"Query Results: {results}")
