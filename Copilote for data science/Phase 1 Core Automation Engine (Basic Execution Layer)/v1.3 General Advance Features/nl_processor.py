import json
from groq import Groq
from schemas import JSON_SCHEMA
from data_handler import EnhancedDataHandler
from fuzzywuzzy import process

class NaturalLanguageProcessor:
    def __init__(self, data_handler: EnhancedDataHandler, api_key: str):
        self.data_handler = data_handler
        self.client = Groq(api_key=api_key)
        self.data_summary = data_handler.get_data_summary()

    def parse_command(self, user_input: str):
        """Convert natural language to structured JSON"""
        prompt = f"""**Dataset Context**
{self.data_summary}

**Command**
{user_input}

Generate JSON using this schema:
{json.dumps(JSON_SCHEMA)}

**Interpretation Guide**
1. For charts: "Show X vs Y" → "x": "X", "y": "Y"
2. For heatmaps: "by A and B" → "rows": "A", "cols": "B"
3. For pie charts: "distribution of Z" → "x": "Z", "y": "count"
4. Colors/styles: "colored by C" → "hue": "C"
5. Aggregations: "average of D" → "operation": "mean", "column": "D"

Return ONLY valid JSON."""
        
        try:
            completion = self.client.chat.completions.create(
                model="qwen-2.5-coder-32b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
                stream=False
            )
            return self._validate_command(json.loads(completion.choices[0].message.content))
        except Exception as e:
            return self._suggest_corrections(user_input, str(e))

    def _validate_command(self, command):
        """Validate and correct command structure"""
        if not isinstance(command, dict):
            return {"error": "Invalid command format"}
        
        # Validate columns in the command
        if "columns" in command:
            valid_columns = self.data_handler.get_columns()
            for col in command["columns"]:
                if col not in valid_columns:
                    match = self.data_handler.find_column_match(col)
                    if not match:
                        return {"error": f"Column '{col}' not found. Similar columns: {self._get_similar_columns(col)}"}
                    command["columns"][command["columns"].index(col)] = match
        
        # Validate conditions
        if "conditions" in command:
            for cond in command["conditions"]:
                col = cond["column"]
                if col not in self.data_handler.get_columns():
                    match = self.data_handler.find_column_match(col)
                    if not match:
                        return {"error": f"Column '{col}' not found. Similar columns: {self._get_similar_columns(col)}"}
                    cond["column"] = match
        
        return command

    def _suggest_corrections(self, user_input, error_message):
        """Suggest valid columns using fuzzy matching"""
        valid_columns = self.data_handler.get_columns()
        invalid_column = error_message.split(":")[1].strip()
        matches = process.extract(invalid_column, valid_columns, limit=5)
        suggestions = [m[0] for m in matches]
        return {"error": f"Invalid column '{invalid_column}'. Did you mean {', '.join(suggestions)}?"}