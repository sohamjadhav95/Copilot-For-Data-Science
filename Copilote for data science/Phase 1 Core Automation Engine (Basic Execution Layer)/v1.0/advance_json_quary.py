import pandas as pd
import sqlite3
import json
from groq import Groq
from fuzzywuzzy import process

class EnhancedDataHandler:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.original_columns = self.df.columns.tolist()
        self.clean_columns = [col.lower().strip() for col in self.original_columns]
        self.conn = sqlite3.connect(":memory:")
        self.df.to_sql("data", self.conn, index=False, if_exists="replace")
        
    def find_best_column_match(self, user_input):
        """Fuzzy match column names with error correction"""
        user_input = user_input.lower().strip()
        match, score = process.extractOne(user_input, self.clean_columns)
        return self.original_columns[self.clean_columns.index(match)] if score > 75 else None

    def get_columns(self):
        return self.original_columns

class NaturalLanguageProcessor:
    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["filter", "aggregate", "sort", "group", "calculate"]},
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

    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.client = Groq(api_key="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")

    def parse_command(self, user_input):
        """Convert natural language to structured JSON with column validation"""
        prompt = f"""Convert this command to JSON: {user_input}
        Use this schema: {json.dumps(self.JSON_SCHEMA)}
        Important: Use NATURAL LANGUAGE COLUMN NAMES from the user's request.
        Return ONLY valid JSON, no other text."""
        
        try:
            completion = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
                stream=False
            )
            raw_json = completion.choices[0].message.content
            return self._validate_and_correct_columns(json.loads(raw_json))
        except Exception as e:
            return {"error": str(e)}

    def _validate_and_correct_columns(self, command):
        """Auto-correct column names using fuzzy matching"""
        corrections = {}
        
        # Validate main columns
        if "columns" in command:
            corrected = []
            for col in command["columns"]:
                match = self.data_handler.find_best_column_match(col)
                if not match:
                    return {"error": f"Column '{col}' not found. Similar columns: {self._get_similar_columns(col)}"}
                corrected.append(match)
                if col != match: corrections[col] = match
            command["columns"] = corrected

        # Validate condition columns
        if "conditions" in command:
            for cond in command["conditions"]:
                match = self.data_handler.find_best_column_match(cond["column"])
                if not match:
                    return {"error": f"Column '{cond['column']}' not found. Similar columns: {self._get_similar_columns(cond['column'])}"}
                if cond["column"] != match: corrections[cond["column"]] = match
                cond["column"] = match

        # Validate aggregation columns
        if "aggregation" in command:
            agg_col = command["aggregation"].get("column")
            if agg_col:
                match = self.data_handler.find_best_column_match(agg_col)
                if not match:
                    return {"error": f"Column '{agg_col}' not found. Similar columns: {self._get_similar_columns(agg_col)}"}
                if agg_col != match: corrections[agg_col] = match
                command["aggregation"]["column"] = match

        if corrections:
            command["column_corrections"] = corrections
            
        return command

    def _get_similar_columns(self, user_input):
        """Get similar column names for error messages"""
        user_input = user_input.lower().strip()
        matches = process.extract(user_input, self.data_handler.clean_columns, limit=3)
        return [self.data_handler.original_columns[self.data_handler.clean_columns.index(m[0])] for m in matches]

class SmartQueryExecutor:
    def __init__(self, data_handler):
        self.df = data_handler.df
        self.conn = data_handler.conn

    def execute(self, command):
        """Execute validated JSON command"""
        if "error" in command:
            return command["error"]
        
        action = command.get("action")
        
        try:
            if action == "filter":
                return self._execute_filter(command)
            elif action == "aggregate":
                return self._execute_aggregation(command)
            elif action == "sort":
                return self._execute_sort(command)
            elif action == "group":
                return self._execute_group(command)
            elif action == "calculate":
                return self._execute_calculation(command)
            else:
                return "Unsupported action"
        except Exception as e:
            return f"Execution error: {str(e)}"

    def _execute_filter(self, command):
        """Handle filtering with auto-corrected columns"""
        conditions = []
        for cond in command.get("conditions", []):
            col = cond["column"]
            op = cond["operator"]
            val = cond["value"]
            
            if op == ">":
                conditions.append(f"`{col}` > {val}")
            elif op == "==":
                conditions.append(f"`{col}` == '{val}'")
                
        query_str = " & ".join(conditions)
        return self.df.query(query_str) if query_str else self.df

    def _execute_aggregation(self, command):
        """Handle max/min/avg aggregations"""
        agg = command["aggregation"]
        col = agg["column"]
        operation = agg["operation"].lower()
        
        if operation == "max":
            result = self.df[self.df[col] == self.df[col].max()]
        elif operation == "min":
            result = self.df[self.df[col] == self.df[col].min()]
        else:
            result = self.df.agg({col: operation})
        
        return result[command["columns"]]

    def _execute_sort(self, command):
        """Handle sorting with multiple columns"""
        ascending = command.get("order", "asc").lower() == "asc"
        return self.df.sort_values(by=command["columns"], ascending=ascending)

    def _execute_group(self, command):
        """Handle grouping and aggregations"""
        grouped = self.df.groupby(command["columns"])
        agg_col = command["aggregation"]["column"]
        operation = command["aggregation"]["operation"].lower()
        return grouped.agg({agg_col: operation}).reset_index()

def main():
    # Initialize with sample data
    data_handler = EnhancedDataHandler(r"C:\Users\soham\Downloads\synthetic_sales_data.csv")
    nl_processor = NaturalLanguageProcessor(data_handler)
    executor = SmartQueryExecutor(data_handler)

    sample_queries = [
        "group by country and show total revenue"
    ]

    for query in sample_queries:
        print(f"\n{'='*50}\nQuery: {query}")
        
        # Process command
        command = nl_processor.parse_command(query)
        print("\nProcessed Command:")
        print(json.dumps(command, indent=2))
        
        # Execute command
        result = executor.execute(command)
        
        print("\nResult:")
        if isinstance(result, pd.DataFrame):
            print(result.to_string(index=False))
        else:
            print(result)

if __name__ == "__main__":
    main()