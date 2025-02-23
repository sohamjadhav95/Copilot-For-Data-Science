import re  # Import for cleaning text
from groq import Groq

# Configure the Groq API with your API key
client = Groq(api_key="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")  # Replace with your Groq API key

def Groq_Input_error(user_input):
    prompt = (
        f"This is an error in data science  message: {user_input}.\n"
        f"Just explain the error in simple words.\n"
    )

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=4096,
        top_p=0.95,
        stream=False,
        stop=None,
    )

    generated_explanation = completion.choices[0].message.content
    print(generated_explanation)


def Groq_Explainer(user_input):
    prompt = (
        f"Generate human-readable explanations for this results: {user_input}.\n"
    )

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=4096,
        top_p=0.95,
        stream=False,
        stop=None,
    )

    generated_explanation = completion.choices[0].message.content
    print(generated_explanation)