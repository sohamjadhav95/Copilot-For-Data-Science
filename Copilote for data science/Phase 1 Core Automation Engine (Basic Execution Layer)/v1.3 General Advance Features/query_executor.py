import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from language_intent_processor import Groq_Input_error
from language_intent_processor import Groq_Explainer

class QueryExecutor:
    def __init__(self, data_handler):
        self.df = data_handler.df
        self.data_handler = data_handler

    def execute(self, command):
        """Execute validated JSON command"""
        try:
            if "error" in command:
                return self._generate_error_explanation(command["error"])
            
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
            elif action == "clean":
                return self._execute_cleanup(command)
            elif action == "explain":
                return self._generate_explanation(command)
            else:
                return "Unsupported action"
        except Exception as e:
            return self._generate_error_explanation(e)

    def _generate_error_explanation(self, error_message):
        return Groq_Input_error(error_message)

    def _generate_explanation(self, command):
        """Generate human-readable explanations for results in real-time"""
        return Groq_Explainer(command)

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


    def _execute_visualization(self, command):
        """Generate data visualizations with natural language support"""
        vis_config = command["visualization"]
        try:
            # Auto-infer missing parameters
            self._infer_visualization_parameters(command)
            
            # Validate required parameters
            required = {
                "heatmap": ["rows", "cols", "value"],
                "pie": ["x"],
                "bar": ["x", "y"],
                "scatter": ["x", "y"]
            }
            
            missing = [param for param in required.get(vis_config["type"], []) 
                       if not vis_config.get(param)]
            if missing:
                return f"Missing parameters for {vis_config['type']}: {', '.join(missing)}"

            # Generate visualization
            if vis_config["type"] == "heatmap":
                pivot = self.df.pivot_table(
                    index=vis_config["rows"],
                    columns=vis_config["cols"],
                    values=vis_config["value"],
                    aggfunc="mean"
                )
                sns.heatmap(pivot, annot=True)
                plt.title(f"{vis_config['value']} by {vis_config['rows']} and {vis_config['cols']}")
                
            elif vis_config["type"] == "pie":
                data = self.df[vis_config["x"]].value_counts()
                data.plot(kind="pie", autopct='%1.1f%%')
                
            elif vis_config["type"] == "scatter":
                sns.scatterplot(
                    data=self.df,
                    x=vis_config["x"],
                    y=vis_config["y"],
                    hue=vis_config.get("hue")
                )
                plt.title(vis_config.get("title", "Scatter Plot"))
                plt.xlabel(vis_config.get("x"))
                plt.ylabel(vis_config.get("y"))
                
            elif vis_config["type"] == "bar":
                data = self.df.groupby(vis_config["x"])[vis_config["y"]].mean()
                if data.empty:
                    return "No data available for bar chart"
                data.plot(kind="bar", color=vis_config.get("color", "skyblue"))
                plt.title(vis_config.get("title", "Bar Chart"))
                plt.xlabel(vis_config.get("x"))
                plt.ylabel(vis_config.get("y"))
                
            plt.show()
            return f"{vis_config['type']} chart displayed"
        except Exception as e:
            return self._generate_error_explanation(str(e))

    def _infer_visualization_parameters(self, command):
        """Infer missing parameters from the original command"""
        vis_config = command["visualization"]
        original_command = command.get("original_command", "")
        
        if vis_config["type"] == "heatmap":
            if "by" in original_command:
                parts = original_command.split("by")[-1].split("and")
                vis_config["rows"] = parts[0].strip()
                vis_config["cols"] = parts[1].strip()
                vis_config["value"] = vis_config.get("value", "Revenue")
        
        if vis_config["type"] == "pie" and not vis_config.get("y"):
            vis_config["y"] = "count"