# queryhandler.py
from intenthandler import get_full_course_title
from feedback import load_llm

def user_is_asking_for_link(query):
    keywords = ["link", "source", "reference", "find", "references", "sources", "url", "where can I find", "can you provide the link"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)

def extract_links_from_source_docs(source_docs):
    links = []
    seen_links = set()  # Set to track unique links

    for doc in source_docs:
        try:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            link = metadata.get("link")
            title = metadata.get("title")

            # Only add unique links
            if link and link not in seen_links:
                links.append((title, link))
                seen_links.add(link)
        except AttributeError:
            # Handle the case where doc.metadata is not accessible
            continue

    return links

async def retrieve_syllabus_documents(course_titles, chroma_store):
    # Build the metadata filter
    metadata_filter = {"title": {"$in": course_titles}}

    # Perform similarity search with the filter
    try:
        results = chroma_store.similarity_search("", k=5, filter=metadata_filter)
        return results
    except Exception as e:
        print(f"Error during syllabus retrieval: {e}")
        return []

async def handle_syllabus_query(query: str, parameters: dict, memory, chroma_store) -> dict:
    # Extract course name from parameters
    course_inputs = parameters.get("Course_name", [])
    if not course_inputs:
        return {"answer":"Please specify the course name or ID for which you want the syllabus."}
    print(course_inputs)
    # Get the full course title from the input
    course_titles = [get_full_course_title(course_input) for course_input in course_inputs]
    print('Here are the course titles: ', course_titles)
    # Retrieve syllabus documents
    retrieved_docs = await retrieve_syllabus_documents(course_titles, chroma_store)
    if not retrieved_docs:
        return {"answer": "Sorry, I couldn't find the syllabus for the specified course."}

    # Prepare the context from the retrieved documents
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    # Construct the prompt for the LLM
    prompt = f"""
    You are a helpful assistant that organizes syllabus information into a clear and concise tabular format. Based on the context provided, generate a week-by-week syllabus for each course mentioned in the query. Use readable and well-structured text in the table.

        Context:{context}
        Question: {query}
        Format the answer as follows:
        | Week | Topic Description |
        |------|-------------------|
        | 1    | Topic details     |
        | 2    | Topic details     |
        ...
        Ensure that:
        1. The information is concise and well-organized.
        2. If a course does not have sufficient details in the context, indicate it with "Details not available."
        Answer:
        """
    # Generate the answer using the LLM
    try:
        llm = load_llm()
        response =  llm.invoke([prompt])
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response)
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return {"response": "An error occurred while generating the syllabus response."}

    return {"answer": response.content, "source_documents": retrieved_docs}

def select_retriever(intent_name, chroma_store, chroma_store1):
    if intent_name == "Get_Research_info":
        return chroma_store1.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    else:
        return chroma_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})
