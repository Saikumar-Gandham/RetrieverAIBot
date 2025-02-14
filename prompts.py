from langchain.prompts import ChatPromptTemplate
def load_prompt(intent_name):
    if intent_name == "Get_Course_info":
        prompt = """You need to answer the user's question about course related and  prerequisites information.
         Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Context: {context}
        Question: {question}"""
    elif intent_name == "Get_CPT_OPT_info":
        prompt = """You need to answer the user's question about CPT or OPT information based on the given context only.
        Give friendly responses.
        Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Context: {context}
        Question: {question}"""
    elif intent_name == "Get_General_info":
        prompt = """You are a helpful assistant for students, here to provide accurate answers strictly based on the provided context.
        Do not create or infer any information that isn’t explicitly in the context. If the answer cannot be found within the context, respond with a polite message indicating that additional information is needed and prompt the student to either rephrase their question or clarify. If applicable, you may suggest specific topics or terms that could help refine the search.
        Context: {context}
        Student's Question: {question}
        Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Example Response (when information is missing):
        I couldn't find a direct answer in the provided information. Could you provide more details or rephrase your question? You might also consider specifying terms or topics to help refine the search."""

    elif intent_name == "Get_Research_info":
        prompt = """You need to answer the user's question about research information. Please dont assume answer only based on the context given
        Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Context: {context}
        Question: {question}"""
    else:
        prompt = """You need to answer the user's question.
        Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Context: {context}
        Question: {question}"""
    return ChatPromptTemplate.from_template(prompt)
