import pandas as pd
from groq import Groq
import json
from fuzzywuzzy import process

class ExcelQueryEngine:
    def __init__(self, file_path, api_key):
        self.df = pd.read_excel(file_path)
        self.client = Groq(api_key=api_key)
    
    def process_nlp_query(self, query):
        """Process the user query using LLM to generate structured JSON"""
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
                "metric": llm_response.get("metric", "Units Sold"),
                "aggregation": llm_response.get("aggregation", "total"),
                "group_by": llm_response.get("group_by", ["Country"]),
                "filters": llm_response.get("filters", {}),
                "sort": llm_response.get("sort", "desc"),
                "limit": llm_response.get("limit", None)
            }
        }
        return structured_query
    
    def execute_query(self, structured_query):
        df_filtered = self.df  # No filters for now

        # Get available columns
        available_columns = df_filtered.columns.tolist()

        # Extract query components
        metric = structured_query.get("query", {}).get("metric")
        aggregation = structured_query.get("query", {}).get("aggregation")
        group_by = structured_query.get("query", {}).get("group_by")

        # Ensure group_by is a list
        if not isinstance(group_by, list):
            group_by = [group_by] if group_by else []

        # 🔍 **Fuzzy match metric to dataset columns**
        metric_match = process.extractOne(metric, available_columns)
        if metric_match and metric_match[1] > 80:
            metric = metric_match[0]
        else:
            return f"❌ Metric '{metric}' not found in dataset!"

        # 🔍 **Fuzzy match group_by to dataset columns**
        group_by_matches = []
        for col in group_by:
            match = process.extractOne(col, available_columns)
            if match and match[1] > 80:
                group_by_matches.append(match[0])

        # If no valid group_by columns found, return an error
        if not group_by_matches:
            return f"❌ Group-by column(s) '{group_by}' not found in dataset!"

        print(f"🔍 Metric Match: {metric}")
        print(f"🔍 Group By Match: {group_by} -> {group_by_matches}")

        # **Apply aggregation**
        try:
            if aggregation == "total":
                result = df_filtered.groupby(group_by_matches)[metric].sum()
            elif aggregation == "average":
                result = df_filtered.groupby(group_by_matches)[metric].mean()
            elif aggregation == "count":
                result = df_filtered.groupby(group_by_matches)[metric].count()
            else:
                return f"❌ Unsupported aggregation type '{aggregation}'!"
        except KeyError as e:
            return f"❌ Column not found: {str(e)}"

        return result.to_dict()

# Set file path and API key
file_path = r"C:\Users\soham\Downloads\synthetic_sales_data.xlsx"
api_key = "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"  # Replace with your actual API key

# Create Engine
engine = ExcelQueryEngine(file_path, api_key)

# Test Queries
user_query = "Show total units sold by product in Germany."
structured_query = engine.process_nlp_query(user_query)
print("Generated Query:", structured_query)

# Execute and print results
results = engine.execute_query(structured_query)
print("Query Results:", results)
