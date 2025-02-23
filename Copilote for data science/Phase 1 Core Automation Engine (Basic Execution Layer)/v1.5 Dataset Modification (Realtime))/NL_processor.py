import re
from groq import Groq

# Configure the Groq API with your API key
client = Groq(api_key="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")  # Replace with your Groq API key

def NL_processor(user_input):
    try:
        # Determine if the user input is for visualization or display
# In NL_processor.py, update the prompt to:
        prompt = (
            f"Determine if the user input requests to 'visualize', 'display', or 'modify' data: {user_input}\n"
            f"Respond ONLY with 'visualize', 'display', or 'modify'."
        )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
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