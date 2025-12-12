from llm_client import client
import vector_store
import mongodb
import json

MODEL = "qwen2.5:1.5b"

def get_routing_decision(history: list, user_question: str) -> str:
    

    # Simple history for the router
    history_str = json.dumps(history[-3:]) # Just last 3 messages
    
    prompt = f"""
    You are a 'router' AI. Your job is to classify the user's latest question.
    The user is in a chat about a document.

    Here is the chat history (in JSON):
    {history_str}

    Here is the new user question:
    "{user_question}"

    Your task is to decide if this question requires searching the document for context.

    Respond with the single word 'search' or 'chat'.

    - Respond 'search' if the question is about the *content* of the document.
      (e.g., 'What is X?', 'Summarize paragraph Y', 'Who is Jane in the document?')

    - Respond 'chat' if the question is *conversational* or *about the chat itself*.
      (e.g., 'Hello', 'Thanks!', 'You are helpful', 'That's wrong', 'Can you repeat that?')

    **CRUCIAL RULE:** A question about the *conversation history* (like 'What was my first question?' or 'What did you just say?') is **ALWAYS** 'chat'.
    """
    
    try:
        response = client.chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        decision = response['message']['content'].strip().lower()
        
        if "search" in decision:
            return "search"
        else:
            return "chat"
            
    except Exception as e:
        print(f"Error in router, defaulting to 'search': {e}")
        return "search" # Default to search if router fails

def answer_from_document(doc_id: int, user_question: str):
    """
    Performs RAG OR simple chat to answer a question.
    """
    
    # 1. Get History
    print("Fetching chat history...")
    history = mongodb.get_chat_history(str(doc_id))
    
    # 2. Get Routing Decision
    print("Routing question...")
    decision = get_routing_decision(history, user_question)
    print(f"Router decision: {decision.upper()}")

    
    messages = []
    
    if decision == "search":
        print(f"Searching document {doc_id} for context...")
        context_chunks = vector_store.search_document(
            doc_id=doc_id, 
            query_text=user_question, 
            top_k=3
        )
        
        if not context_chunks:
            return "I couldn't find any relevant information in that document to answer your question."
        
        context = "\n\n---\n\n".join(context_chunks)
        
        system_prompt = """You are an assistant for 'intelliDocs'. Your task is to answer questions based ONLY on the provided context.Do not use any outside knowledge. If the answer is not in the context, state that clearly."""
        
        # Start building the messages
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]
        
        messages.extend(history)
        
        # Add the final user prompt WITH context
        final_user_prompt = f"""
        CONTEXT:
        {context}

        QUESTION:
        {user_question}
        """
        messages.append({'role': 'user', 'content': final_user_prompt})

    else: 
        print("Answering as a chatbot...")
        
        system_prompt = """You are 'intelliDocs', a helpful AI assistant. The user is asking a conversational question. Answer them based on the provided chat history in third person. Be friendly and direct."""
        
        # Start building the messages
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]
        
        # Add history
        messages.extend(history)
        
        # Add the final user prompt WITHOUT context
        messages.append({'role': 'user', 'content': user_question})

    # 4. Call the Ollama model
    try:
        print(f"\n... Sending final prompt to {MODEL} ...\n")
        response = client.chat(
            model=MODEL,
            messages=messages
        )
        
        return response['message']['content']
        
    except Exception as e:
        print(f"Error contacting Ollama: {e}")
        return "An error occurred while trying to get an answer from the model."