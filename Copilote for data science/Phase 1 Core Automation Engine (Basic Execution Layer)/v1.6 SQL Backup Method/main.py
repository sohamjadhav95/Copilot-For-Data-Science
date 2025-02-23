from NL_processor import NL_processor
from Engine_Display_data import Groq_Input as Display_Groq_Input
from Engine_Visualize_data import Groq_Input as Visualize_Groq_Input
from Engine_Modify_data import Groq_Input as Modify_Groq_Input

def main():
    while True:
        user_input = input("Enter your request (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Determine if the request is to visualize or display data
        operation = NL_processor(user_input)

        if operation == "visualize":
            print("Visualizing data...")
            Visualize_Groq_Input(user_input)
        elif operation == "display":
            print("Displaying data...")
            Display_Groq_Input(user_input)
        elif operation == "modify":
            print("Modifying data...")
            Modify_Groq_Input(user_input)
        else:
            print("Unable to determine the operation. Please try again.")

if __name__ == "__main__":
    main()