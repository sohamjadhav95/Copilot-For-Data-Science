import re
from groq import Groq
from Data import dataset_features
from config.api_manager import get_api_key
# Configure the Groq API with your API key
client = Groq(api_key=get_api_key())  # Replace with your Groq API key

def NL_processor(user_input):
    try:
        # Determine if the user input is for visualization or display
        prompt = f"""
        Analyze the user input and determine its intent based on the following categories:

        1. **Data Operations:** If the input requests to 'visualize', 'display', or 'modify' or 'undo' data.
        2. **General Inquiry:** If the input is asking for general information or a meaningful response.
        3. **Machine Learning Tasks:**
        - 'build' → Building an ML model using AutoML.
        - 'predict' or 'deploy' → Deploying an ML model.
        - 'test' → Testing an ML model.
        - 'predict custom input' → Predicting custom input.
        4. **Data Analysis & Reporting:**
        - 'analyze' → Performing data analysis.
        - 'report' → Generating a data report.
        - 'dashboard' → Creating a dashboard.

        User Input: "{user_input}"

        ### Expected Response:
        Respond **ONLY** with one of the following categories based on the input:
        - 'visualize' (for data visualization)
        - 'display' (for displaying data)
        - 'modify' (for modifying data)
        - 'undo' (for undoing last modification)
        - 'meaningful_response' (for general information or meaningful answers)
        - 'build_model' (for ML model building)
        - 'deploy_model' (for ML model deployment or prediction)
        - 'test_model' (for ML model testing)
        - 'predict_custom_input' (for predicting custom input from user)
        - 'analyze_data' (for data analysis)
        - 'generate_report' (for report generation)
        - 'create_dashboard' (for dashboard creation)
        """

        completion = client.chat.completions.create(
            model="gemma2-9b-it",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=4096,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        response = completion.choices[0].message.content.strip().lower()

        if "visualize" in response:
            return "visualize"
        elif "display" in response:
            return "display"
        elif "modify" in response:
            return "modify"
        elif "undo" in response:
            return "undo"
        elif "meaningful_response" in response:
            return "meaningful_response"
        elif "build_model" in response:
            return "build_model"
        elif "deploy_model" in response:
            return "deploy_model"
        elif "test_model" in response:
            return "test_model"
        elif "predict_custom_input" in response:
            return "predict_custom_input"
        elif "analyze_data" in response:
            return "analyze_data"
        elif "generate_report" in response:
            return "generate_report"
        elif "create_dashboard" in response:
            return "create_dashboard"
        else:
            return None
    except Exception as e:
        print(f"An error occurred in NL_processor: {e}")
        return None


def split_multi_commands(user_input):
    """
    Use NLP to split the user input into individual commands.
    """
    try:
        # Prompt the Groq API to split the input into commands
        prompt = (
            f"Split the following input into individual when necessary.\n"
            f"commands: {user_input}\n"
            f"Respond ONLY with the commands separated by '||'. For example: "
            "'Show me first 10 rows of the dataset||Visualize the main insight of data||Clean all null value rows from the whole dataset'"
        )

        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        response = completion.choices[0].message.content.strip()
        
        # Split the response into commands using the delimiter '||'
        commands = response.split("||")
        commands = [cmd.strip() for cmd in commands if cmd.strip()]  # Remove empty commands
        return commands
    except Exception as e:
        print(f"Error in split_multi_commands: {e}")
        return [user_input]  # Fallback to treating the entire input as a single command


def result_response(user_input ,result):
    '''
    This Genertes a response based on the result provided
    '''

    prompt = (
        f"Generate a meaningful response for user input: {user_input}, and result is executed: {result}\n"
        f"You can also suggest something based on after execution of result\n"
        f"Respond ONLY with one or two sentence.\n"
    )

    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1024,
        top_p=0.95,
        stream=False,
        stop=None,
    )

    response = completion.choices[0].message.content.strip()
    print(response)


def genral_response_chatbot(user_input):
    '''
    This function generates a response to the user input using the Chatbot model
    '''
    prompt = (
        f"Generate a meaningful response for user input: {user_input}\n"
        f"Refer this dataset features if you need to: {dataset_features()}, 'otherwise avoid it'\n"
        f"Respond in Brief.\n"
    )

    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=1024,
        top_p=0.95,
        stream=False,
        stop=None,
    )

    response = completion.choices[0].message.content.strip()
    print(response)


if __name__ == "__main__":
    user_input = "Undo last modification"
    print(NL_processor(user_input))
