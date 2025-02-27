import pandas as pd
from Data import filepath
import dash
from dash import dcc, html
import plotly.express as px
from groq import Groq
import webbrowser
import pandas as pd
from fpdf import FPDF
import json
import re

client = Groq(api_key="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")  # Replace with your Groq API key


def load_and_preprocess_data():
    data_path = filepath()
    df = pd.read_csv(data_path)
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)  # Example: Forward fill missing values
    
    # Convert categorical data if necessary
    # df['column_name'] = pd.Categorical(df['column_name'])
    
    return df


class Dashboard:
    def __init__(self):
        pass
        
    def create_dashboard(self, user_input=None):
        df = self.load_and_preprocess_data()
        app = dash.Dash(__name__)
        
        # Generate JSON config from natural language input
        if user_input:
            config = self.generate_dashboard_config(user_input, df.columns.tolist())
        else:
            config = self.default_dashboard_config(df.columns.tolist())
        
        # Validate and parse the configuration
        try:
            validated_config = self.validate_config(config, df.columns)
        except ValueError as e:
            print(f"Invalid configuration: {e}")
            return

        # Create layout from config
        app.layout = self.build_layout_from_config(validated_config, df)
        
        webbrowser.open('http://127.0.0.1:8050/')
        app.run_server(debug=True)

    def generate_dashboard_config(self, user_input, available_columns):
        """Generate JSON config using Qwen-2.5-coder-32b model"""
        
        prompt = f"""
        Convert this dashboard request to JSON configuration. Available columns: {available_columns}
        
        Example Response:
        {{
            "num_charts": 2,
            "charts": [
                {{"type": "Bar", "x": "Date", "y": "Sales"}},
                {{"type": "Scatter", "x": "Revenue", "y": "Cost"}}
            ]
        }}
        
        User Request: "{user_input}"
        
        Respond ONLY with valid JSON. Ensure column names exactly match: {available_columns}
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        
        raw_response = response.choices[0].message.content
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        try:
            print("API Response:", response.choices[0].message.content)  # Log the raw response
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response from API: {e}")
            return self.default_dashboard_config(available_columns)

    def validate_config(self, config, df_columns):
        """Validate the configuration against dataset columns"""
        required_keys = {"num_charts", "charts"}
        chart_keys = {"type", "x", "y"}
        
        if not all(key in config for key in required_keys):
            raise ValueError("Missing required configuration keys")
        
        if len(config["charts"]) != config["num_charts"]:
            raise ValueError("Number of charts doesn't match configuration")
        
        for chart in config["charts"]:
            if not all(key in chart for key in chart_keys):
                raise ValueError(f"Missing keys in chart configuration: {chart}")
            if chart["x"] not in df_columns or chart["y"] not in df_columns:
                raise ValueError(f"Invalid column names in chart: {chart}")
        
        return config

    def build_layout_from_config(self, config, df):
        """Build Dash layout from validated configuration"""
        charts = []
        
        for idx, chart_config in enumerate(config["charts"]):
            fig = self.create_figure(df, chart_config)
            charts.append(
                html.Div([
                    html.H3(f"Chart {idx+1}: {chart_config['type']}"),
                    dcc.Graph(figure=fig)
                ], style={'border': '1px solid #ddd', 'padding': '10px', 'margin': '10px'})
            )
        
        return html.Div([
            html.H1("Automated Data Dashboard", style={'textAlign': 'center'}),
            html.Div(charts),
            html.Div(id='insights-output')
        ])

    def create_figure(self, df, chart_config):
        """Create plotly figure based on chart configuration"""
        chart_type = chart_config["type"].lower()
        x_col = chart_config["x"]
        y_col = chart_config["y"]
        
        if chart_type == "bar":
            return px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        elif chart_type == "scatter":
            return px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        elif chart_type == "histogram":
            return px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
        elif chart_type == "line":
            return px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        else:
            return px.scatter(df, x=x_col, y=y_col)  # Default fallback

    def default_dashboard_config(self, columns):
        """Fallback configuration if API fails"""
        return {
            "num_charts": 2,
            "charts": [
                {"type": "Histogram", "x": columns[0], "y": columns[1]},
                {"type": "Scatter", "x": columns[2], "y": columns[3]}
            ]
        }

    def load_and_preprocess_data(self):
        data_path = filepath()  # Use filepath directly
        df = pd.read_csv(data_path)  # Use data_path instead of self.data_path
        
        # Handle missing values
        df.fillna(method='ffill', inplace=True)  # Example: Forward fill missing values
        
        # Convert categorical data if necessary
        # df['column_name'] = pd.Categorical(df['column_name'])
        
        return df


def analyze_data(user_input):
    try:
        df = load_and_preprocess_data()
        insights = generate_insights()
        print("Generated Insights:")
        print(insights)
        generate_pdf_report(df, insights)
        create_dashboard()
    except Exception as e:
        print(f"An error occurred during data analysis: {e}")


def generate_pdf_report(df, insights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Data Analysis Report", ln=True, align='C')
    pdf.cell(200, 10, txt="Summary Statistics", ln=True, align='L')
    pdf.multi_cell(0, 10, txt=df.describe().to_string())
    
    pdf.cell(200, 10, txt="AI-Generated Insights", ln=True, align='L')
    pdf.multi_cell(0, 10, txt=insights)
    
    pdf.output("report.pdf")



if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.create_dashboard("Show me house value vs ocean proximity in bar plot")