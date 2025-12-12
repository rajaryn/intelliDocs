import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

client = None
db = None
chat_history_collection = None

try:
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "intellidocs")

    if not MONGO_URI:
        raise Exception("MONGO_URI not found in .env file. Make sure your .env file is correct.")

    # Create the client and connect
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    
    # Check the connection
    client.admin.command('ping')
    
    # Get the database
    db = client[MONGO_DB_NAME]
    
    chat_history_collection = db["chat_histories"] 

    print("Successfully connected to MongoDB Atlas.")

except ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
    print("Please check your MONGO_URI, network access, or Atlas IP whitelist.")
    # sys.exit(1) # Optionally exit if the DB is required
except Exception as e:
    print(f"An error occurred during DB initialization: {e}")
# --- End of Connection ---


def save_message_to_history(session_id: str, role: str, content: str):
    """
    Saves a single message (from user or assistant) to a chat history.
    
    If the session_id doesn't exist, it creates a new document.
    If it does exist, it appends the message to the 'messages' array.
    """
    try:
        # The message to be added
        message = {"role": role, "content": content}
        
        # Find document by session_id and add the message to the 'messages' array
        # 'upsert=True' creates the document if it doesn't exist.
        chat_history_collection.update_one(
            {"session_id": session_id},
            {"$push": {"messages": message}},
            upsert=True
        )
        print(f"Saved message for session: {session_id}")
    except Exception as e:
        print(f"Error saving message: {e}")

def get_chat_history(session_id: str) -> list:
    """
    Retrieves the full chat history (the 'messages' array) for a session.
    Returns an empty list if the session is not found.
    """
    try:
        history_doc = chat_history_collection.find_one(
            {"session_id": session_id}
        )
        
        if history_doc:
            # Return just the 'messages' array, or [] if it doesn't exist
            return history_doc.get("messages", [])
        else:
            # No history found for this session
            return []
            
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return []
    

def delete_chat_history(session_id: str):
    """
    Deletes all chat messages associated with a specific session(document) ID.
    """
    try:
        result = chat_history_collection.delete_many({"session_id": session_id})
        print(f"Deleted {result.deleted_count} chat messages for {session_id} from MongoDB.")
        return True
    except Exception as e:
        print(f"Error deleting from MongoDB: {e}")
        return False