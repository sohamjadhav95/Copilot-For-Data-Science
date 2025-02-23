import pandas as pd
import sqlite3
import json
import numpy as np
from groq import Groq
from fuzzywuzzy import process

class EnhancedDataHandler:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.original_columns = self.df.columns.tolist()
        self.clean_columns = [col.lower().strip() for col in self.original_columns]
        self.conn = sqlite3.connect(":memory:")
        self.df.to_sql("data", self.conn, index=False, if_exists="replace")
        
    def get_data_summary(self):
        """Create a detailed summary for the AI model"""
        summary = "Dataset Structure:\n"
        summary += f"Number of Rows: {len(self.df)}\n"
        summary += f"Number of Columns: {len(self.original_columns)}\n\n"
        summary += "Columns Details:\n"
        
        for col in self.original_columns:
            dtype = str(self.df[col].dtype)
            unique = self.df[col].nunique()
            sample = ", ".join(map(str, self.df[col].dropna().head(3).tolist()))
            summary += f"- {col} ({dtype}): {unique} unique values\n  Sample: {sample}\n"
            
            # Add statistical summaries for numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                summary += f"  Stats: Mean={self.df[col].mean():.2f}, Min={self.df[col].min():.2f}, Max={self.df[col].max():.2f}\n"
            
            # Add ASCII histogram for numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                hist, bins = np.histogram(self.df[col].dropna(), bins=5)
                hist_ascii = "".join(["▇" if h > 0 else " " for h in hist])
                summary += f"  Distribution: {hist_ascii}\n"
        
        return summary

    def find_best_column_match(self, user_input):
        """Fuzzy match column names with error correction"""
        user_input = user_input.lower().strip()
        match, score = process.extractOne(user_input, self.clean_columns)
        return self.original_columns[self.clean_columns.index(match)] if score > 75 else None

class NaturalLanguageProcessor:
    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["filter", "aggregate", "sort", "group", "calculate", "add_column"]},
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
            },
            "add_column": {
                "type": "object",
                "properties": {
                    "new_column": {"type": "string"},
                    "formula": {"type": "string"}
                }
            }
        }
    }

    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.client = Groq(api_key="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")
        self.data_summary = data_handler.get_data_summary()

    def parse_command(self, user_input):
        """Convert natural language to structured JSON with data context"""
        prompt = f"""**Dataset Context**
{self.data_summary}

**User Command**
{user_input}

Convert this command to JSON using the schema below. Consider:
- Use "group" action for aggregation tasks
- Actual column names from dataset context
- Appropriate data types for operations
- Valid operations for column types

Schema: {json.dumps(self.JSON_SCHEMA)}
Return ONLY valid JSON, no other text."""
        
        try:
            completion = self.client.chat.completions.create(
                model="qwen-2.5-coder-32b",
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
        if not isinstance(command, dict):
            return {"error": "Invalid command format"}
        
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
        self.data_handler = data_handler

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
                return self._execute_aggregation(command)  # Reuse aggregation logic
            elif action == "calculate":
                return self._execute_calculation(command)
            elif action == "add_column":
                return self._execute_add_column(command)
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
        """Handle aggregation with grouping and filtering"""
        try:
            # Apply conditions if present
            df_filtered = self.df
            if "conditions" in command:
                conditions = []
                for cond in command["conditions"]:
                    col = cond["column"]
                    op = cond["operator"]
                    val = cond["value"]
                    
                    if op == ">":
                        conditions.append(f"`{col}` > {val}")
                    elif op == "==":
                        conditions.append(f"`{col}` == '{val}'")
                        
                query_str = " & ".join(conditions)
                df_filtered = self.df.query(query_str) if query_str else self.df

            # Perform aggregation
            agg = command["aggregation"]
            col = agg["column"]
            operation = agg["operation"].lower()
            
            if "columns" in command:
                grouped = df_filtered.groupby(command["columns"])
                result = grouped.agg({col: operation}).reset_index()
            else:
                result = df_filtered.agg({col: operation})
            
            return result
        except Exception as e:
            return f"Aggregation error: {str(e)}"

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

    def _execute_add_column(self, command):
        """Add a new column based on a formula"""
        add_col = command["add_column"]
        new_col = add_col["new_column"]
        formula = add_col["formula"]
        
        # Evaluate the formula dynamically
        self.df[new_col] = self.df.eval(formula)
        return self.df    
    # Existing implementation remains unchanged

def main():
    csv_file = r"C:\Users\soham\Downloads\synthetic_sales_data.csv"
    data_handler = EnhancedDataHandler(csv_file)
    nl_processor = NaturalLanguageProcessor(data_handler)
    executor = SmartQueryExecutor(data_handler)

    print("="*50)
    print("Initial Dataset Summary:")
    print(data_handler.get_data_summary())

    while True:
        user_command = input("\nEnter command (or 'exit'): ").strip()
        if user_command.lower() == "exit":
            break

        # Process command with data context
        command = nl_processor.parse_command(user_command)
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