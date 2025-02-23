import pandas as pd
import sqlite3
import json
from groq import Groq

# Initialize Groq API client
api_key = "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"
client = Groq(api_key=api_key)

# JSON schema template for the AI response
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["filter", "aggregate", "sort", "group"]},
        "columns": {"type": "array", "items": {"type": "string"}},
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operator": {"type": "string"},
                    "value": {"type": ["number", "string"]}
                }
            }
        },
        "aggregation": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "operation": {"type": "string"}
            }
        }
    }
}

class DataHandler:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.conn = sqlite3.connect(":memory:")
        self.df.to_sql("data", self.conn, index=False, if_exists="replace")
        
    def get_columns(self):
        return self.df.columns.tolist()

def parse_command(command):
    """Convert natural language command to structured JSON"""
    prompt = f"""Convert this command to JSON: {command}
    Use this schema: {JSON_SCHEMA}
    Return ONLY valid JSON, no other text."""
    
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
        stream=False
    )
    return completion.choices[0].message.content

def parse_json_response(response):
    """Parse and validate JSON response"""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}
    except Exception as e:
        return {"error": str(e)}

class QueryExecutor:
    def __init__(self, data_handler):
        self.df = data_handler.df
        self.conn = data_handler.conn
    
    def execute(self, query_json):
        """Execute command based on JSON structure"""
        if "error" in query_json:
            return query_json["error"]
        
        action = query_json.get("action", "")
        
        if action == "filter":
            return self._handle_filter(query_json)
        elif action == "aggregate":
            return self._handle_aggregation(query_json)
        elif action == "sort":
            return self._handle_sort(query_json)
        else:
            return "Unsupported action"

    def _handle_filter(self, query):
        """Handle filtering operations"""
        try:
            conditions = []
            for cond in query.get("conditions", []):
                col = cond["column"]
                op = cond["operator"]
                val = cond["value"]
                
                if op == ">":
                    conditions.append(f"{col} > {val}")
                elif op == "==":
                    conditions.append(f"{col} == '{val}'")
                
            query_str = " & ".join(conditions)
            return self.df.query(query_str) if query_str else self.df
            
        except Exception as e:
            return f"Filter error: {str(e)}"

    def _handle_aggregation(self, query):
        """Handle aggregation operations"""
        try:
            agg = query.get("aggregation", {})
            col = agg.get("column", "")
            operation = agg.get("operation", "").lower()
            
            if operation == "max":
                max_row = self.df[self.df[col] == self.df[col].max()]
                return max_row[query.get("columns", [])]
            
            return f"Unsupported aggregation: {operation}"
        except Exception as e:
            return f"Aggregation error: {str(e)}"

def main():
    csv_file = r"C:\Users\soham\Downloads\synthetic_sales_data.csv"
    data_handler = DataHandler(csv_file)
    executor = QueryExecutor(data_handler)

    user_command = "filter where Units Sold > 4900"

    # Get and parse JSON command
    json_response = parse_command(user_command)
    parsed_command = parse_json_response(json_response)
    
    print("\nParsed Command:")
    print(json.dumps(parsed_command, indent=2))
    
    # Execute command
    result = executor.execute(parsed_command)
    print("\nResult:")
    print(result if isinstance(result, str) else result.to_string(index=False))

if __name__ == "__main__":
    main()