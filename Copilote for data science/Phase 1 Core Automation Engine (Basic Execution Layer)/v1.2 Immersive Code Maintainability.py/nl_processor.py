import json
from groq import Groq
from schemas import JSON_SCHEMA
from data_handler import EnhancedDataHandler

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
Focus on valid column names and operations."""
        
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
            return {"error": str(e)}

    def _validate_command(self, command):
        """Validate and correct command structure"""
        # Validation logic here
        return command