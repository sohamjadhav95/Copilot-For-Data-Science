from data_handler import EnhancedDataHandler
from nl_processor import NaturalLanguageProcessor
from query_executor import QueryExecutor
import pandas as pd

def main():
    # Configuration
    CSV_FILE = r"C:\Users\soham\Downloads\synthetic_sales_data.csv"
    API_KEY = "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"
    
    # Initialize components
    data_handler = EnhancedDataHandler(CSV_FILE)
    nl_processor = NaturalLanguageProcessor(data_handler, API_KEY)
    executor = QueryExecutor(data_handler)
    
    print("="*50)
    print(data_handler.get_data_summary())
    
    while True:
        command = input("\nEnter command (or 'exit'): ").strip()
        if command.lower() == "exit":
            break
        
        # Process and execute command
        json_command = nl_processor.parse_command(command)
        result = executor.execute(json_command)
        
        # Display results
        if isinstance(result, pd.DataFrame):
            print(result.to_string(index=False))
        else:
            print(result)

if __name__ == "__main__":
    main()