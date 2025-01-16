from groq import Groq
from datetime import datetime
current_datetime = datetime.now()
current_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
import json

f = open('creds.json')
data = json.load(f)
api_key = data["groq_api"]
f.close()

client = Groq(
        api_key=api_key,
    )
def QuestionClassification(input_text):

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                # "content": "You are an assistant which classifies question whether it is open-ended, closed-ended, rhetorical, reflective, leading or clarifying."
                "content": "You are an assistant which classifies question in one word whether it is factual,external or self ?"
            },
            {
                "role": "user",
                "content": input_text
            }
            
        ],
        max_tokens=10,
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content
    # return chat_completion
    
    
def generate_google_search_query(input_text):


    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant which generates google search query based on user query? In this format: What is today's news about adani, result : today+news+adani. please dont return anything else"
            },
            {
                "role": "user",
                "content": input_text
            }
            
        ],
        max_tokens=100,
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content


def generate_content(input_text):


    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Your name is Alexa. you are an assistant which provides a short answer to user question",
            },
            {
                "role": "system",
                "content": "today's date and time is "+current_datetime,
            },
            {
                "role": "user",
                "content": input_text,
            }
            
        ],
        max_tokens=200,
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content


def generate_content_context(context,input_text):

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Your name is SKY.",
            },
            {
                "role": "system",
                "content": "You are an assistant who will provide the answer based on context and user query. If the context is too big just summarize the contents in short.",
            },
            {
                "role": "system",
                "content": "today's date and time is "+current_datetime,
            },
            {
                "role": "user",
                "content": "<context>"+context+"\n"+"<query>"+input_text,
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

def is_image_generation_required(input_text):
    routing_prompt = f"""
    Given the following input text, determine whether it requires image generation. respond with yes or no
    for example : user prompt : How are you ? assistant : no, user prompt : What do you think of the universe ? assistant : yes
    User query: {input_text}
    Response:
    """
    
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are an assistant which determines if the image generation is required for the user prompt."},
            {"role": "user", "content": routing_prompt}
        ],
        max_tokens=10  # We only need a short response
    )

    image_required = response.choices[0].message.content.strip()
    print(image_required)
    return image_required
    