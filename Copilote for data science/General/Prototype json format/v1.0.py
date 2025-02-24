import pandas as pd
import matplotlib.pyplot as plt

class ExcelQueryEngine:
    def __init__(self, file_path):
        """Load Excel file into a Pandas DataFrame."""
        self.df = pd.read_excel(file_path)
    
    def execute(self, query):
        """Process a structured query and execute the corresponding operation."""
        operation = query.get("operation")
        parameters = query.get("parameters", {})
        
        if hasattr(self, operation):
            return getattr(self, operation)(**parameters)
        else:
            return f"Unsupported operation: {operation}"
    
    def filter(self, column, condition, value):
        """Filter rows based on a condition (>, <, ==)."""
        if condition == ">":
            return self.df[self.df[column] > value]
        elif condition == "<":
            return self.df[self.df[column] < value]
        elif condition == "==":
            return self.df[self.df[column] == value]
        return self.df
    
    def sort(self, columns, ascending=True):
        """Sort DataFrame by specified columns."""
        return self.df.sort_values(by=columns, ascending=ascending)
    
    def aggregate(self, group_by, agg_column, agg_func):
        """Apply aggregation function like sum, mean, etc."""
        return self.df.groupby(group_by)[agg_column].agg(agg_func)
    
    def pivot_table(self, index, columns, values, aggfunc):
        """Create a pivot table."""
        return self.df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
    
    def visualize(self, chart_type, x, y, aggfunc):
        """Generate charts based on data."""
        data = self.df.groupby(x)[y].agg(aggfunc)
        data.plot(kind=chart_type)
        plt.show()

# Example Usage
df_engine = ExcelQueryEngine(r"C:\Users\soham\Downloads\Financial Sample.xlsx")
query = {
    "operation": "aggregate",
    "parameters": {
        "group_by": ["Country"],
        "agg_column": "Units Sold",
        "agg_func": "sum"
    }
}
result = df_engine.execute(query)
print(result)
