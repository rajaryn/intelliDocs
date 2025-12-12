from llm_client import client

def generate_tags_for_text(text: str) -> list[str]:
    max_text_length = 8000
    truncated_text = text[:max_text_length]

    prompt = f"""
    Analyze the following document text and generate between 5 and 7 relevant, single-word or two-word, lowercase tags that categorize the content.
    Return these tags as a single, comma-separated string ONLY. Do not provide any explanation, preamble, or markdown formatting.

    Example Response: machine_learning,python,data_science,neural_networks,research

    Document Text:
    ---
    {truncated_text}
    ---
    """
    
    try:
        # Call the local Ollama API. By default, stream=False.
        response = client.chat(
            model='qwen2.5:1.5b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            options={
                'temperature': 0.1 # Low temperature for more predictable, structured output.
            }
        )
        
        # Parse the complete response from Ollama.
        tags_string = response.message.content.strip().lower()

        # Clean up the string and split it into a list.
        tags_list = [tag.strip() for tag in tags_string.split(',') if tag.strip()]
        
        #print(f"Ollama Generated Tags: {tags_list}")
        return tags_list

    except Exception as e:
        print(f"Error communicating with Ollama for tagging: {e}")
        return []


def generate_summary_for_text(text: str) -> str:
    """
    Uses a local Ollama model to generate a summary for a given text.
    Waits for the full response before returning.
    """
    # Use a larger portion of the text for a better summary.
    max_text_length = 16000
    truncated_text = text[:max_text_length]

    # This prompt asks for a natural language paragraph.
    prompt = f"""
    Analyze the provided document text and generate a concise summary.

    **Instructions:**
   - The summary must be a single paragraph  of about 100-150 words or it can be of two paragraphs at max for total word limit to be 400 words .
   - It is crucial that you provide ONLY the summary text itself.
   - DO NOT include any titles, preambles like "Summary:", or concluding remarks.

   **Example:**
   ---
   Document Text: "The sun is a star at the center of the Solar System. It is a nearly perfect sphere of hot plasma. Earth and other matter orbit it."
   Summary: The sun, a star at the center of our Solar System, is a hot plasma sphere orbited by Earth and other bodies.
   ---

   **Document to Summarize:**
   ---
  {truncated_text}
  ---
  """
    
    try:
        # Call the local Ollama API.
        response = client.chat(
            model='qwen2.5:1.5b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            options={
                'temperature': 0.4 # Slightly higher temperature for more creative/natural summary writing.
            }
        )
        
        # Parse the complete summary from the response.
        summary = response['message']['content'].strip()
        #print(f"Ollama Generated Summary: {summary[:120]}...")
        return summary

    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""