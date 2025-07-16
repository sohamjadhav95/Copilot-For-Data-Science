
# Copilot for Data Science

## Overview
This project, "Copilot for Data Science," aims to provide an interactive and automated platform for data analysis, visualization, and machine learning tasks. It leverages natural language processing (NLP) to interpret user commands and perform various data operations, including data display, modification, visualization, and machine learning model building and deployment.

## Features
- **Natural Language Processing (NLP)**: Interpret user commands in natural language.
- **Data Operations**: Display, modify, and visualize data.
- **Machine Learning**: Build, test, deploy, and predict using machine learning models.
- **Dashboard Creation**: Generate interactive dashboards for data visualization.
- **Automated Error Handling**: Provide dynamic error explanations and solutions.
- **Data Analysis and Reporting**: Generate insights and PDF reports from datasets.

## Installation

### Prerequisites
- Python 3.8 or higher
- Basic knowledge of Python and machine learning concepts

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Copilot-For-Data-Science.git
   cd Copilot-For-Data-Science

### Install Dependencies:

pip install -r requirements.txt

### Set Up API Keys:
Ensure you have the necessary API keys for Groq and other services.
Place the API keys in the appropriate configuration files or environment variables.

### Prepare the Dataset:
Place your dataset in the data directory.
Update the Data.py file with the correct path to your dataset.

### Usage
Running the Application

### Start the Application:
python main.py

### Interact with the Application:
Use natural language commands to perform various operations. For example:
- Show me the first 10 rows of the dataset
- Visualize the main insight of data
- Clean all null value rows from the whole dataset
- Build a machine learning model
- Deploy the model and predict custom input

### Example Commands
Display Data:
- Show me the first 10 rows of the dataset
Modify Data:
- Clean all null value rows from the whole dataset
Visualize Data:
- Visualize the main insight of data
Machine Learning:
- Build a machine learning model
- Deploy the model and predict custom input

### Directory Structure
```
Copilot-For-Data-Science/
├── main.py
|
├── Retrival_Agumented_Generation/
|   ├── rag_command_parser.py
|   └── commands_database.csv
|
├── Machine_Learning/
|   └── ML_Models_Engine_autogluon.py
|
├── Core_Automation_Engine/
|   ├── Engine_Data_analysis.py
|   ├── Engine_Display_data.py
|   ├── Engine_Modify_data.py
|   ├── Engine_Visualize_data.py
|   ├── NL_processor.py
|   └── Data.py
|
├── requirements.txt
└── README.md
```
### Contributing
Contributions are welcome! Please follow these guidelines:

Fork the repository.
Create a new branch for your feature or bug fix.
Commit your changes and push to your fork.
Submit a pull request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contact
For any questions or issues, please contact:
soham.ai.engineer@gmail.com or omkargadakh1272004@gmail.com

## v3.5 Deployment: Major New Features

The v3.5 Deployment introduces a powerful new command-line interface (CLI) and advanced automation features for data science workflows:

- **Rich CLI Interface**: Use `rich_cli.py` for an interactive, colorized command-line experience. Launch with `python rich_cli.py` or use the provided `copilot.bat`.
- **Multi-Command Support**: Chain multiple commands in a single input using "then", "and", or "after" (e.g., "Show data then visualize main insight").
- **Automated Dashboards**: Instantly generate interactive dashboards using Dash and Plotly, with smart configuration from natural language.
- **Machine Learning Automation**: Build, test, deploy, and predict with ML models using AutoML (AutoGluon, PyCaret, TPOT, etc.).
- **PDF Report Generation**: Automatically create PDF reports from your data analysis.
- **Advanced Error Handling**: Dynamic, AI-powered explanations and solutions for errors encountered during operations.
- **Enhanced Data Operations**: Display, modify, visualize, and analyze data with improved NLP and context awareness.

### How to Use the v3.5 CLI

1. **Install dependencies** (from the v3.5 Deployment folder):
   ```bash
   pip install -r "Copilote for data science/Phase 3 Human in the Loop (Interactive AI)/v3.5 Deployment/requirements.txt"
   ```
2. **Launch the CLI**:
   - Windows: Double-click `copilot.bat` in the v3.5 Deployment folder, or run:
     ```bash
     python rich_cli.py
     ```
3. **Interact with the CLI**:
   - Type natural language commands (single or chained):
     - `Show me the first 10 rows of the dataset`
     - `Visualize the main insight of data and then build a machine learning model`
     - `Clean all null value rows then generate a PDF report`
   - Use `exit` to quit.

### Requirements
See `Copilote for data science/Phase 3 Human in the Loop (Interactive AI)/v3.5 Deployment/requirements.txt` for the full list of dependencies.
