import os
import chromadb
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time

# --- CONFIGURATION SWITCH ---
USE_PINECONE = True 

# Pinecone Configuration
PINECONE_INDEX_NAME = "intellidocs"
PINECONE_DIMENSION = 384 # Matches 'all-MiniLM-L6-v2'
PINECONE_METRIC = "cosine"

# --- INITIALIZATION ---
print("Loading embedding model...")
# This runs locally on the server (CPU) for both options
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

# Global variables for the chosen client
pinecone_index = None
chroma_collection = None

def get_vector_client():
    """
    Initializes connection to either Pinecone or ChromaDB based on the switch.
    """
    global pinecone_index, chroma_collection

    if USE_PINECONE:
        # --- OPTION A: PINECONE (CLOUD) ---
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("ERROR: PINECONE_API_KEY not found in environment variables.")
            return

        print(f"Connecting to Pinecone Index: {PINECONE_INDEX_NAME}...")
        try:
            pc = Pinecone(api_key=api_key)
            
            # Check if index exists, if not create it (Serverless)
            existing_indexes = [i.name for i in pc.list_indexes()]
            if PINECONE_INDEX_NAME not in existing_indexes:
                print(f"   Index not found. Creating '{PINECONE_INDEX_NAME}'...")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric=PINECONE_METRIC,
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait a moment for initialization
                time.sleep(10)
            
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            print("Pinecone Connected.")
            
        except Exception as e:
            print(f"Failed to connect to Pinecone: {e}")

    else:
        # --- OPTION B: CHROMA (LOCAL) ---
        print("Initializing Local ChromaDB...")
        try:
            client = chromadb.PersistentClient(path="./chroma_db")
            chroma_collection = client.get_or_create_collection(name="documents")
            print("ChromaDB Ready.")
        except Exception as e:
            print(f"Failed to initialize ChromaDB: {e}")

# Initialize the client immediately on module import
get_vector_client()


# --- MAIN FUNCTIONS (Unified Interface) ---

def add_document_chunks(doc_id: int, chunks: list[str]):
    """
    Creates embeddings and adds them to the selected Vector Store (Pinecone or Chroma).
    """
    if not chunks:
        print(f"No chunks provided for doc_id {doc_id}.")
        return

    print(f"Generating {len(chunks)} embeddings for doc_id {doc_id}...")
    
    try:
        # 1. Generate Embeddings (Same for both)
        embeddings = EMBEDDING_MODEL.encode(chunks).tolist()

        # --- PATH A: PINECONE ---
        if USE_PINECONE and pinecone_index:
            vectors_to_upsert = []
            for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                # Pinecone expects: (id, vector_values, metadata)
                vector_id = f"{doc_id}_{i}"
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": vector,
                    "metadata": {
                        "doc_id": str(doc_id), # Store as string for filtering
                        "text": chunk
                    }
                })
            
            # Upsert in batches of 100 to avoid request limits
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                pinecone_index.upsert(vectors=batch)
            
            print(f"Uploaded {len(chunks)} chunks to Pinecone.")

        # --- PATH B: CHROMA ---
        elif not USE_PINECONE and chroma_collection:
            metadatas = [{'doc_id': str(doc_id)} for _ in chunks]
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            
            chroma_collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Saved {len(chunks)} chunks to ChromaDB.")
            
        else:
            print("Error: No valid vector store client initialized.")

    except Exception as e:
        print(f"Error adding document chunks: {e}")


def search_document(doc_id: int, query_text: str, top_k: int = 5) -> list:
    """
    Searches for relevant chunks in the selected Vector Store.
    """
    try:
        # 1. Generate Query Embedding
        query_embedding = EMBEDDING_MODEL.encode([query_text]).tolist()
        
        # --- PATH A: PINECONE ---
        if USE_PINECONE and pinecone_index:
            # Flatten list for Pinecone (it expects a single list of floats, not list of lists)
            flat_embedding = query_embedding[0]
            
            results = pinecone_index.query(
                vector=flat_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={
                    "doc_id": {"$eq": str(doc_id)} # Filter by doc_id
                }
            )
            
            # Extract text from metadata
            return [match['metadata']['text'] for match in results['matches']]

        # --- PATH B: CHROMA ---
        elif not USE_PINECONE and chroma_collection:
            results = chroma_collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where={"doc_id": str(doc_id)}
            )
            return results['documents'][0] if results['documents'] else []

        else:
            print("Error: No valid vector store client initialized.")
            return []

    except Exception as e:
        print(f"Error during search: {e}")
        return []