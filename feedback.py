# feedback.py

import aiofiles
import json
from langchain.schema import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI

# Feedback variables
affirmative_responses = [
    "yes", "yup", "yeah", "absolutely", "definitely", "sure",
    "yes, this is helpful", "it's helpful", "very helpful", "indeed",
    "positive", "certainly", "of course", "thank you", "this is great", "glad"
]

negative_responses = [
    "no", "nope", "not really", "not at all", "negative",
    "unfortunately not", "it's not helpful", "no, not helpful",
    "no thanks", "nah", "not quite", "don't think so", "disappointed", "bad"
]

async def save_conversation_history(conversation_data):
    # Process the chat_history to make it JSON serializable
    processed_chat_history = []
    for message in conversation_data['chat_history']:
        if isinstance(message, HumanMessage):
            processed_chat_history.append({
                'type': 'human',
                'content': message.content
            })
        elif isinstance(message, AIMessage):
            processed_chat_history.append({
                'type': 'ai',
                'content': message.content
            })
        else:
            # Handle other message types if necessary
            processed_chat_history.append({
                'type': 'other',
                'content': message.content
            })
    # Replace the chat_history with the processed version
    conversation_data['chat_history'] = processed_chat_history

    # Now it's safe to serialize conversation_data to JSON
    async with aiofiles.open("conversation_history.json", mode="a") as file:
        await file.write(json.dumps(conversation_data) + "\n")

def get_last_updated_date(log_file_path):
    import re
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
        for line in reversed(lines):
            match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
            if match:
                return match.group(0)
        return "Unknown"
    except Exception as e:
        print(f"Error reading log file: {e}")
        return "Unknown"

def load_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)