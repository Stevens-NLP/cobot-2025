import requests


def get_messages():
    # Define the Canvas API URL
    canvas_instance = "sit.instructure.com"  # Replace with your Canvas domain
    url = f"https://{canvas_instance}/api/v1/conversations"

    # Define query parameters
    params = {
        "scope": "unread",  # Options: unread, starred, archived, sent (remove this for all messages)
    }


    # Define headers with the access token
    headers = {
        "Authorization": "Bearer 1030~eKh6WrTYtmUJyeaemXTPR6DPNFJ6vyY8LUGRvfW7Q8NEa3ZDtJ7vK4aCKvmGc6VX"
    }

    # Make the GET request
    response = requests.get(url, headers=headers, params=params)
    response =response.json()
    messages = {"subject":[],"message":[],"sender_name":[],"last_message_at":[]}
    num_messages = len(response)
    for res in response:
        subject = res["subject"]
        message = res["last_message"]
        sender_name = res["participants"][0]["name"]
        last_message_at = res["last_message_at"]
        messages["subject"].append(subject)
        messages["message"].append(message)
        messages["sender_name"].append(sender_name)
        messages["last_message_at"].append(last_message_at)
    return messages,len(messages["message"])