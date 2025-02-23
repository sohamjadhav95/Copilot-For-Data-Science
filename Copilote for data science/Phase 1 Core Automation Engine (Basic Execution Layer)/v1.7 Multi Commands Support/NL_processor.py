import re
from groq import Groq

# Configure the Groq API with your API key
client = Groq(api_key="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")  # Replace with your Groq API key

def split_multi_commands(user_input):
    """
    Use NLP to split the user input into individual commands.
    """
    try:
        # Prompt the Groq API to split the input into commands
        prompt = (
            f"Split the following input into individual commands: {user_input}\n"
            "Respond ONLY with the commands separated by '||'. For example: "
            "'Show me first 10 rows of the dataset||Visualize the main insight of data||Clean all null value rows from the whole dataset'"
        )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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


def NL_processor(user_input):
    try:
        # Determine if the user input is for visualization or display
        prompt = (
            f"Determine if the user input requests to 'visualize', 'display', or 'modify' data: {user_input}\n"
            f"Respond ONLY with 'visualize', 'display', or 'modify'."
        )

        completion = client.chat.completions.create(
            model="qwen-2.5-32b",  
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
        else:
            return None
    except Exception as e:
        print(f"An error occurred in NL_processor: {e}")
        return None


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
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1024,
        top_p=0.95,
        stream=False,
        stop=None,
    )

    response = completion.choices[0].message.content.strip()
    print(response)

if __name__ == "__main__":
    user_input = "Display the first 10 rows of the dataset"
    result = NL_processor(user_input)
    print(result)

