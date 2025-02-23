import pandas as pd
import matplotlib.pyplot as plt

class QueryExecutor:
    def __init__(self, data_handler):
        self.df = data_handler.df
        self.data_handler = data_handler

    def execute(self, command):
        """Execute validated JSON command"""
        try:
            if "error" in command:
                return command["error"]
            
            action = command.get("action")
            
            if action == "filter":
                return self._execute_filter(command)
            elif action == "group":
                return self._execute_group(command)
            elif action == "visualize":
                return self._execute_visualization(command)
            elif action == "calculate":
                return self._execute_calculation(command)
            elif action == "add_column":
                return self._execute_add_column(command)
            elif action == "aggregate":
                return self._execute_aggregation(command)
            elif action == "sort":
                return self._execute_sort(command)
            # Add other actions
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

    def _execute_group(self, command):
        """Handle grouping and aggregations"""
        grouped = self.df.groupby(command["columns"])
        agg_col = command["aggregation"]["column"]
        operation = command["aggregation"]["operation"].lower()
        return grouped.agg({agg_col: operation}).reset_index()

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

    def _execute_add_column(self, command):
        """Add a new column based on a formula"""
        add_col = command["add_column"]
        new_col = add_col["new_column"]
        formula = add_col["formula"]
        
        # Evaluate the formula dynamically
        self.df[new_col] = self.df.eval(formula)
        return self.df    
    # Existing implementation remains unchanged

    def _execute_visualization(self, command):
        """Generate data visualizations"""
        try:
            vis_config = command["visualization"]
            fig, ax = plt.subplots()
            
            if vis_config["type"] == "bar":
                data = self.df.groupby(vis_config["x"])[vis_config["y"]].mean()
                data.plot(kind="bar", ax=ax)
                ax.set_title(vis_config.get("title", "Bar Chart"))
            elif vis_config["type"] == "line":
                data = self.df.groupby(vis_config["x"])[vis_config["y"]].mean()
                data.plot(kind="line", ax=ax)
                ax.set_title(vis_config.get("title", "Line Chart"))
            elif vis_config["type"] == "pie":
                data = self.df.groupby(vis_config["x"])[vis_config["y"]].sum()
                data.plot(kind="pie", autopct='%1.1f%%', ax=ax)
                ax.set_title(vis_config.get("title", "Pie Chart"))
            else:
                return "Unsupported visualization type"
            
            plt.show()
            return f"{vis_config['type'].capitalize()} chart displayed"
        except Exception as e:
            return f"Visualization error: {str(e)}"