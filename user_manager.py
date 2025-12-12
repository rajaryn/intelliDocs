import cloudinary
import cloudinary.uploader
from pinecone import Pinecone
import os
from database import get_db_connection

# Import your MongoDB helper
import mongodb 

# --- CONFIGURATION ---
cloudinary.config(
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key = os.getenv('CLOUDINARY_API_KEY'),
    api_secret = os.getenv('CLOUDINARY_API_SECRET')
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Check if index exists before connecting to avoid errors
if "intellidocs" in [i.name for i in pc.list_indexes()]:
    index = pc.Index("intellidocs")
else:
    index = None

def delete_user_account(user_id):
    """
    Wipes User from: SQL, Cloudinary, Pinecone, and MongoDB.
    """
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed"

    try:
        cursor = conn.cursor(dictionary=True)

        print(f"STARTING ACCOUNT DELETION FOR USER {user_id} ")

        # 1. FETCH METADATA (Crucial Step)
        # We need the list of documents belonging to this user BEFORE we delete the user.
        cursor.execute("SELECT id, public_id FROM documents WHERE user_id = %s", (user_id,))
        documents = cursor.fetchall()
        
        # Extract lists for batch operations
        doc_ids = [str(doc['id']) for doc in documents]         # For Pinecone & Mongo
        public_ids = [doc['public_id'] for doc in documents]    # For Cloudinary

        if not doc_ids:
            print("   User has no documents. Proceeding to delete user record.")
        else:
            print(f"   Found {len(doc_ids)} documents to clean up.")

            # 2. DELETE FROM CLOUDINARY (Files)
            if public_ids:
                print(f"   Deleting files from Cloudinary...")
                cloudinary.api.delete_resources(public_ids)

            # 3. DELETE FROM PINECONE (Vectors)
            if index:
                print(f"   Deleting vectors from Pinecone...")
                for d_id in doc_ids:
                    # Delete all vectors where metadata['doc_id'] == d_id
                    try:
                        index.delete(filter={"doc_id": {"$eq": d_id}})
                    except Exception as p_err:
                        print(f"      Warning: Pinecone delete failed for doc {d_id}: {p_err}")

            # 4. DELETE FROM MONGODB (Chat History)
            print(f"   Deleting chat history from MongoDB...")
            for d_id in doc_ids:
                mongodb.delete_chat_history(d_id)

        # 5. DELETE FROM SQL (TiDB)
        # This is the final blow. ON DELETE CASCADE will remove the rows 
        # from the 'documents' table automatically.
        print(f"   Deleting user record from SQL...")
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()

        print("✅ Account deletion complete.")
        return True, "Account successfully deleted."

    except Exception as e:
        print(f"RITICAL ERROR deleting account: {e}")
        conn.rollback()
        return False, str(e)
    finally:
        if cursor: cursor.close()
        if conn: conn.close()