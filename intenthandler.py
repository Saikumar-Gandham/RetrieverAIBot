# intenthandler.py

def get_parent_intent(intent_name):
    """
    Returns the parent intent for a given intent. If the intent is a parent intent, returns itself.
    """
    # Intent mapping dictionary
    intent_mapping = {
        "Get_Course_info": {
            "sub_intents": [
                "Get_Course_info - custom"
            ]
        },
        "Get_CPT_OPT_Info": {
            "sub_intents": [
                "Get_CPT_Application_Process",
                "Get_CPT_OPT_Documents",
                "Get_OPT_Application_Process",
                "Get_CPT_Eligibility",
                "Get_OPT_Eligibility"
            ]
        },
        "Get_General_Info": {
            "sub_intents": [
                "Get_General_Info - custom"
            ]
        },
        "Get_Research_Info": {
            "sub_intents": [
                "Get_Research_Faculty_Info",
                "Get_Research_Info - custom"
            ]
        }
    }

    # Iterate through the mapping to find the parent intent
    for parent_intent, details in intent_mapping.items():
        if intent_name == parent_intent or intent_name in details["sub_intents"]:
            return parent_intent

    # If no match is found, return None
    return intent_name

def handle_intent_change(intent_name, memory, feedback_vars):
    """
    Handles changes in user intent and manages interaction counts.
    Sets feedback_requested flag when intent changes or interaction count exceeds threshold.
    """
    last_intent = feedback_vars['last_intent']
    interaction_count = feedback_vars['interaction_count']
    feedback_requested = feedback_vars['feedback_requested']
    pending_conversation_data = feedback_vars['pending_conversation_data']
    feedback_timeout_counter = feedback_vars['feedback_timeout_counter']

    if last_intent is not None and last_intent != intent_name:
        # Intent has changed
        feedback_timeout_counter = 0  # Reset the feedback timeout counter
        # Store the previous conversation data
        pending_conversation_data = {
            "intent": last_intent,
            "chat_history": memory.chat_memory.messages.copy(),
            "feedback": None
        }
        interaction_count = 0  # Reset interaction count for new intent
    else:
        # Same intent, increment interaction count
        interaction_count += 1
        if interaction_count >= 5:
            feedback_requested = True
            feedback_timeout_counter = 0  # Reset the feedback timeout counter
            # Store the current conversation data
            pending_conversation_data = {
                "intent": intent_name,
                "chat_history": memory.chat_memory.messages.copy(),
                "feedback": None
            }
            interaction_count = 0  # Reset after requesting feedback

    last_intent = intent_name  # Update to the new intent

    # Update the feedback_vars dictionary
    feedback_vars.update({
        'last_intent': last_intent,
        'interaction_count': interaction_count,
        'feedback_requested': feedback_requested,
        'pending_conversation_data': pending_conversation_data,
        'feedback_timeout_counter': feedback_timeout_counter
    })

def is_syllabus_query(query: str) -> bool:
    """
    Determines if the query is related to syllabus information.
    """
    syllabus_keywords = ["syllabus", "course outline", "course structure", "course syllabus", "course outcomes"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in syllabus_keywords)


def get_full_course_title(course_input):
    course_mapping = {
  "data 601": "DATA 601 Introduction to Data Science",
  "data 602": "DATA 602 Introduction to Data Analysis and Machine Learning",
  "data 603": "DATA 603 Platforms for Big Data Processing",
  "data 604": "DATA 604 Data Management",
  "data 605": "DATA 605 Ethical Legal Issues in Data Science",
  "data 606": "DATA 606 Capstone in Data Science",
  "data 607": "DATA 607 Leadership in Data Science",
  "data 608": "DATA608 Probability and Statistics for Data Science",
  "data 690": "DATA 690 Special Topics: Statistical Analysis and Visualization with Python",
  "data 690": "DATA 690 Special Topics: Mathematical Foundations for Machine Learning",
  "data 690": "DATA 690 Special Topics: Data Structures and Algorithms in Python",
  "data 690": "DATA 690 Special Topics: Applied Machine Learning with MATLAB",
  "data 690": "DATA 690 Special Topics: Designing Data Driven Web Applications",
  "data 690": "DATA 690 Financial Data Science",
  "data 690": "DATA 690 Special Topic: Modern Practical Deep Learning",
  "data 690": "DATA 690 Special Topics: Introduction to Natural Language Processing",
  "data 690": "DATA 690 Special Topics: Artificial Intelligence for Practitioners",
  "data 696": "DATA 696 Independent Study for Interns and Co-op Students",
  "data 699": "DATA 699 Independent Study in Data Science",
  "data 613": "DATA 613  Data Visualization and Communication",
  "data 621": "DATA 621  Practical Deep Learning"}
    # Normalize input
    course_input_lower = course_input.lower()

    # First, try exact match in keys (course IDs)
    for course_id in course_mapping.keys():
        if course_id.lower() == course_input_lower:
            return course_mapping[course_id]

    # Then, try exact match in values (course titles)
    for course_title in course_mapping.values():
        if course_title.lower() == course_input_lower:
            return course_title

    # No match found
    return None