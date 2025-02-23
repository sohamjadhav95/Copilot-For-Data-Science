import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
import json
from fuzzywuzzy import process
from tpot import TPOTRegressor, TPOTClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Union, List, Dict, Optional

class ExcelQueryEngine:
    def __init__(self, file_path: str, api_key: str):
        """
        Initialize with automated data cleaning and schema analysis.
        """
        self.df = pd.read_excel(file_path)
        self.client = Groq(api_key=api_key)
        self._clean_data()  # Auto-clean on initialization
        self._analyze_schema()
        
        self.allowed_operations = {
            'aggregations': ['sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var'],
            'ml_tasks': ['predict', 'classify', 'forecast'],
            'visualizations': ['bar', 'line', 'histogram', 'scatter']
        }

    def _analyze_schema(self):
        """Auto-detect column types and data issues."""
        self.schema = {
            'numeric_cols': self.df.select_dtypes(include=np.number).columns.tolist(),
            'categorical_cols': self.df.select_dtypes(include='object').columns.tolist(),
            'date_cols': self.df.select_dtypes(include='datetime').columns.tolist(),
            'missing_values': self.df.isnull().sum().to_dict()
        }

    def _clean_data(self):
        """Automated data cleaning pipeline."""
        # Handle missing values
        self.df = self.df.dropna(axis=1, thresh=0.7*len(self.df))  # Drop empty cols
        for col in self.df.columns:
            if self.df[col].dtype in [np.float64, np.int64]:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif self.df[col].dtype == 'object':
                self.df[col].fillna('Unknown', inplace=True)
        
        # Remove outliers using IQR
        for col in self.schema['numeric_cols']:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            self.df = self.df[~((self.df[col] < (q1 - 1.5 * iqr)) | 
                              (self.df[col] > (q3 + 1.5 * iqr)))]

    def process_nlp_query(self, query: str) -> Dict:
        """
        Handles both legacy and advanced queries.
        """
        system_prompt = """
        You are a data science assistant. Convert queries into JSON with:
        - For legacy queries: metrics, group_by, filters
        - For advanced tasks: analysis_type, predict, visualize
        Examples:
        
        Legacy Query ("Show total sales by region for 2023"):
        {
          "metrics": [["Sales", "sum"]],
          "group_by": "Region",
          "filters": [
            {"column": "Date", "condition": ">=", "value": "2023-01-01"}
          ]
        }
        
        Advanced Query ("Predict sales and visualize trends"):
        {
          "analysis_type": ["predict", "visualize"],
          "predict": {"target": "Sales", "problem_type": "regression"},
          "visualize": {"type": "line", "x": "Date", "y": "Sales"}
        }
        """

    def _auto_ml(self, target: str, problem_type: str = 'regression'):
        """Automated machine learning pipeline."""
        try:
            X = self.df.drop(target, axis=1)
            y = self.df[target]

            if problem_type == 'regression':
                model = TPOTRegressor(generations=3, population_size=15)
            else:
                model = TPOTClassifier(generations=3, population_size=15)
                y = LabelEncoder().fit_transform(y)

            model.fit(X, y)
            return model.export()
        except Exception as e:
            return {"error": f"AutoML failed: {str(e)}"}

    def _generate_visualization(self, config: Dict):
        """Automated visualization engine."""
        plt.figure(figsize=(10, 6))
        viz_type = config.get('type', 'bar')
        
        try:
            if viz_type == 'bar':
                sns.barplot(data=self.df, x=config['x'], y=config['y'])
            elif viz_type == 'line':
                sns.lineplot(data=self.df, x=config['x'], y=config['y'])
            elif viz_type == 'histogram':
                sns.histplot(self.df[config['x']])
            elif viz_type == 'scatter':
                sns.scatterplot(data=self.df, x=config['x'], y=config['y'])
            
            plt.title(f"{viz_type.title()} Chart: {config['x']} vs {config.get('y', '')}")
            plt.savefig("auto_plot.png")
            return "Visualization saved as auto_plot.png"
        except Exception as e:
            return {"error": f"Visualization failed: {str(e)}"}

    def execute_query(self, structured_query: Dict) -> Dict:
        """
        Unified execution engine for all query types.
        """
        results = {}
        df = self.df.copy()
        
        try:
            # 1. Always clean data first
            if 'clean' in structured_query.get('analysis_type', []):
                self._clean_data()
                results['cleaning_report'] = self.schema['missing_values']
                df = self.df  # Use cleaned data

            # 2. Apply filters (works for both legacy and new queries)
            if 'filters' in structured_query:
                df = self.apply_filters(df, structured_query['filters'])

            # 3. Handle legacy aggregations
            if 'metrics' in structured_query:
                metrics = [
                    (self._fuzzy_match_column(m[0]), self.allowed_operations['aggregations'].get(m[1].lower()))
                    for m in structured_query['metrics']
                ]
                valid_metrics = [(col, agg) for col, agg in metrics if col and agg]

                if 'group_by' in structured_query:
                    group_col = self._fuzzy_match_column(structured_query['group_by'])
                    if group_col:
                        grouped = df.groupby(group_col)
                        results['aggregations'] = {
                            f"{agg}_{col}": grouped[col].agg(agg)
                            for col, agg in valid_metrics
                        }
                        results['aggregations'] = pd.DataFrame(results['aggregations']).reset_index().to_dict(orient='records')
                else:
                    results['aggregations'] = {
                        f"{agg}_{col}": df[col].agg(agg)
                        for col, agg in valid_metrics
                    }

            # 4. Handle advanced tasks
            if 'predict' in structured_query.get('analysis_type', []):
                results['prediction'] = self._auto_ml(
                    structured_query['predict']['target'],
                    structured_query['predict'].get('problem_type', 'regression')
                )

            if 'visualize' in structured_query.get('analysis_type', []):
                results['visualization'] = self._generate_visualization(
                    structured_query['visualize']
                )

            return results

        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}

# Example Usage
if __name__ == "__main__":
    engine = ExcelQueryEngine(r"C:\Users\soham\Downloads\synthetic_sales_data.xlsx", "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")
    
    complex_query = """
    Clean the data, predict future sales, and show a bar chart of sales by region.
    Filter for dates after 2023.
    """
    
    structured_query = engine.process_nlp_query(complex_query)
    print("Structured Query:", json.dumps(structured_query, indent=2))
    
    results = engine.execute_query(structured_query)
    print("Results:", results)