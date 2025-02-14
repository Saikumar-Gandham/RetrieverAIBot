# main.py

from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import nest_asyncio
import uvicorn
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Import custom modules
from intenthandler import get_parent_intent, handle_intent_change, is_syllabus_query
from queryhandler import user_is_asking_for_link, extract_links_from_source_docs, handle_syllabus_query, select_retriever
from feedback import save_conversation_history, affirmative_responses, negative_responses, get_last_updated_date, load_llm
from prompts import load_prompt

# Initialize FastAPI app
app = FastAPI()

# Load the .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("openai_api_key")
ngrok_auth = os.getenv("ngrok_auth")

# Define persistence directories for Chroma
persist_directory_retriever = "chroma_store"
persist_directory_research = "chroma_store1"

# Initialize OpenAI embeddings and Chroma vector stores
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
chroma_store = Chroma(
    collection_name="retriever_bot",
    embedding_function=embedding_model,
    persist_directory=persist_directory_retriever
)
chroma_store1 = Chroma(
    collection_name="research_info",
    embedding_function=embedding_model,
    persist_directory=persist_directory_research
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)

# Initialize feedback variables
feedback_vars = {
    'last_intent': None,
    'interaction_count': 0,
    'feedback_requested': False,
    'pending_conversation_data': None,
    'feedback_timeout_counter': 0,
    'MAX_FEEDBACK_PROMPTS': 2
}

def get_user_messages(memory):
    user_messages = []
    for msg in memory.chat_memory.messages:
        if msg.type == 'human':
            user_messages.append(msg.content)
    return '\n'.join(user_messages)

@app.post("/webhook")
async def webhook(request: Request):
    global memory, feedback_vars
    # Extract the query and intent from DialogFlow request
    req = await request.json()
    query = req.get("queryResult", {}).get("queryText", "")
    intent_name = req.get("queryResult", {}).get("intent", {}).get("displayName", "")
    parameters = req.get("queryResult", {}).get("parameters", {})

    # Handle the Default Welcome Intent
    if intent_name == "Default Welcome Intent":
        memory.chat_memory.clear()

        log_file_path = "./scraping_log.log"
        last_updated_date = get_last_updated_date(log_file_path)
        response_text = (
                f"Hello! I was last updated on {last_updated_date}. "
                "While I strive to provide accurate and up-to-date details, there may have been changes since this date. How may I assist you today?"
            )
        return JSONResponse(content={"fulfillmentText": response_text})

    intent_name = get_parent_intent(intent_name)
    print(f"Received query: {query} | Intent: {intent_name}")

    # Check if the user has provided feedback
    # **Step 1: Handle Feedback If Previously Requested**
    if feedback_vars['feedback_requested'] and feedback_vars['pending_conversation_data']:
        user_input = query.lower().strip().replace("'", "").replace("\"", "")
        feedback = None
        if user_input in affirmative_responses:
            feedback = "positive"
        elif user_input in negative_responses:
            feedback = "negative"
        else:
            # User has not provided feedback yet
            # Proceed to handle the user's query but keep prompting for feedback
            pass  # We'll handle this below

        if feedback is not None:
            # Update the pending conversation data with feedback
            feedback_vars['pending_conversation_data']["feedback"] = feedback
            # Save the conversation history with feedback
            await save_conversation_history(feedback_vars['pending_conversation_data'])
            # Reset flags and clear memory
            feedback_vars['feedback_requested'] = False
            feedback_vars['pending_conversation_data'] = None
            feedback_vars['feedback_timeout_counter'] = 0  # Reset the feedback timeout counter
            memory.chat_memory.clear()
            response_text = "Thank you for your feedback!"
            print("Feedback received and conversation history saved.")
            return {"fulfillmentText": response_text}
        else:
            # User has not provided valid feedback yet
            feedback_vars['feedback_timeout_counter'] += 1
            if feedback_vars['feedback_timeout_counter'] >= feedback_vars['MAX_FEEDBACK_PROMPTS']:
                # Give up on prompting for feedback after max attempts
                # Save the conversation history without feedback
                await save_conversation_history(feedback_vars['pending_conversation_data'])
                feedback_vars['feedback_requested'] = False
                feedback_vars['pending_conversation_data'] = None
                feedback_vars['feedback_timeout_counter'] = 0
                memory.chat_memory.clear()
                print("Feedback not provided after maximum attempts. Proceeding without feedback.")
                # Proceed to handle the current query as normal
            else:
                pass

    # **Step 2: Handle Intent Change and Interaction Count**
    handle_intent_change(intent_name, memory, feedback_vars)
    # Determine if the query is a syllabus query
    if intent_name == "Get_Course_info" and is_syllabus_query(query):
        # Handle syllabus query separately
        result = await handle_syllabus_query(query, parameters, memory, chroma_store)
        response_text = result['answer']
    else:
        # Select the appropriate retriever and prompt based on intent
        retriever = select_retriever(intent_name, chroma_store, chroma_store1)
        prompt = load_prompt(intent_name)

        # Initialize the question generation and document response chains
        qa_chain = ConversationalRetrievalChain.from_llm(
                llm= load_llm(),
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={'prompt': prompt},
                return_source_documents=True,
                verbose=True,
                rephrase_question=True,
        )
        # Run the query through the QA chain and get the response
        result = qa_chain({"question": query, "chat_history": get_user_messages(memory)})
        response_text = result['answer']

    # Extract source documents
    source_docs = result.get('source_documents', [])

    # Check if the user is asking for the link
    if user_is_asking_for_link(query):
        # Extract links from source documents
        links = extract_links_from_source_docs(source_docs)  # Returns list of URLs or list of (title, URL) tuples
        if links:
            # Append links to the response using Markdown
            response_text += "\n\nHere are the links to the sources:\n\n"

            for link in links:
                if isinstance(link, tuple):
                    title, url = link
                    response_text += f"- [{title}]({url})\n"
                else:
                    response_text += f"- [{link}]({link})\n"
        else:
            response_text += "\n\nSorry, I couldn't find any links to provide."

    # Append the feedback prompt if feedback is requested
    if feedback_vars['feedback_requested']:
        response_text += "\n\nWas this conversation helpful? Please reply with 'Yes' or 'No'."

    # Prepare DialogFlow response
    response = {"fulfillmentText": response_text}

    return response

# Run the FastAPI app using nest_asyncio
nest_asyncio.apply()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
