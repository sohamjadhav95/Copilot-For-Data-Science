import pandas as pd
from groq import Groq
import json
from fuzzywuzzy import process
from typing import Union, List, Dict

class ExcelQueryEngine:
    def __init__(self, file_path: str, api_key: str):
        """
        Initialize the query engine with an Excel file and Groq API key.
        """
        self.df = pd.read_excel(file_path)
        self.client = Groq(api_key=api_key)
        self.allowed_aggregations = {
            'sum': 'sum',
            'average': 'mean',
            'count': 'count',
            'min': 'min',
            'max': 'max',
            'median': 'median',
            'variance': 'var',
            'standard deviation': 'std'
        }

        # Convert date columns to datetime format
        for col in self.df.columns:
            if "date" in col.lower() or "time" in col.lower() or "year" in col.lower():
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    def process_nlp_query(self, query: str) -> Dict:
        """
        Convert natural language query into structured JSON format using NLP.
        """
        system_prompt = """
        You are an expert at converting natural language queries into structured JSON for Excel analysis.
        The dataset contains these columns: Sales, Profit, Region, Product, Date.
        Always return valid JSON with: metrics, group_by, and filters.
        DO NOT use Markdown formatting (e.g., ```json```). Return plain JSON only.
        Example:
        {
          "metrics": [["Sales", "sum"], ["Profit", "average"]],
          "group_by": "Region",
          "filters": [
            {"column": "Date", "condition": ">=", "value": "2023-01-01"},
            {"column": "Product", "condition": "contains", "value": "Electronics"}
          ]
        }
        """
        try:
            completion = self.client.chat.completions.create(
                model="qwen-2.5-32b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                top_p=0.95
            )

            response_text = completion.choices[0].message.content.strip()
            return json.loads(response_text)
        except Exception as e:
            return {"error": f"Query processing failed: {str(e)}"}

    def _fuzzy_match_column(self, user_column: str) -> Union[str, None]:
        """
        Fuzzy match a user-provided column name to dataset columns.
        """
        if pd.isna(user_column) or user_column.strip() == "":
            return None
            
        match, score = process.extractOne(user_column, self.df.columns)
        if score > 70:
            return match
        print(f"⚠️ Column '{user_column}' not found (score: {score})")
        return None

    def _parse_filter_condition(self, condition: str) -> str:
        """
        Map natural language operators to pandas-compatible symbols.
        """
        operator_map = {
            'greater than': '>', 'less than': '<',
            'equal to': '==', 'not equal to': '!=',
            '>=': '>=', '<=': '<=', 'contains': 'str.contains'
        }
        return operator_map.get(condition.lower(), '==')

    def apply_filters(self, df: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
        """
        Apply advanced filters to DataFrame.
        """
        filtered_df = df.copy()
        
        for filter_item in filters:
            col = self._fuzzy_match_column(filter_item['column'])
            if not col:
                continue

            condition = self._parse_filter_condition(filter_item['condition'])
            value = filter_item['value']

            # Handle data types
            try:
                if filtered_df[col].dtype == 'datetime64[ns]':
                    value = pd.to_datetime(value)
                elif filtered_df[col].dtype.kind in 'fiu':
                    value = pd.to_numeric(value, errors='coerce')
            except:
                pass

            # Apply filter
            if condition == 'str.contains':
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(value, na=False)]
            else:
                filtered_df = filtered_df.query(f"`{col}` {condition} @value", local_dict={'value': value})

        return filtered_df

    def execute_query(self, structured_query: Dict) -> Dict:
        """
        Execute structured query from JSON.
        """
        try:
            df = self.df.copy()
            
            # Apply filters
            if 'filters' in structured_query:
                df = self.apply_filters(df, structured_query['filters'])

            # Process metrics and grouping
            results = {}
            if 'metrics' in structured_query:
                metrics = [
                    (self._fuzzy_match_column(m[0]), self.allowed_aggregations.get(m[1].lower()))
                    for m in structured_query['metrics']
                ]
                valid_metrics = [(col, agg) for col, agg in metrics if col and agg]

                if 'group_by' in structured_query:
                    group_col = self._fuzzy_match_column(structured_query['group_by'])
                    if group_col:
                        grouped = df.groupby(group_col)
                        results = {
                            f"{agg}_{col}": grouped[col].agg(agg)
                            for col, agg in valid_metrics
                        }
                        results = pd.DataFrame(results).reset_index().to_dict(orient='records')
                else:
                    results = {
                        f"{agg}_{col}": df[col].agg(agg)
                        for col, agg in valid_metrics
                    }

            return results or {"error": "No valid operations could be performed"}

        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}

# Example Usage
if __name__ == "__main__":
    file_path = r"C:\Users\soham\Downloads\synthetic_sales_data.xlsx"
    api_key = "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"

    engine = ExcelQueryEngine(file_path, api_key)

    user_query = "Show total sales by region for 2023, filtered for electronics products"
    structured_query = engine.process_nlp_query(user_query)
    print("Structured Query:", structured_query)

    results = engine.execute_query(structured_query)
    print("Query Results:", results)