from ollama import Client

# --- CONFIGURATION SWITCH ---
USE_TUNNEL = False 

# Ngrok URL 
TUNNEL_URL = "https://blamably-fatalistic-fritz.ngrok-free.dev"

def get_client():
    if USE_TUNNEL:
        print(f"Connecting to Tunneled Ollama: {TUNNEL_URL}")
        # This part is crucial to bypass ngrok's warning page
        return Client(
            host=TUNNEL_URL, 
            headers={"ngrok-skip-browser-warning": "true"}
        )
    else:
        # print("Connecting to Local Ollama")
        return Client(host='http://localhost:11434')

# Create a shared client instance
client = get_client()