from typing import List
from openai import OpenAI
import os

# Initialize the OpenAI client with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  # Store API key in environment variable
)

#COMBINES CONTEXTS + CONSTRUCT PROMPT + CALL LLM
def call_llm(query: str, contexts: List[str]) -> str:
    """
    OpenRouter API LLM call using google/gemini-2.5-flash model.
    Uses provided contexts to answer the question.
    """

#Combines contexts into single string
    context_text = "\n\n".join(contexts) 

#Construct prompt 
    prompt = f"""You are a helpful assistant.
Use ONLY the context below to answer the question.
Context:
{context_text}

Question:
{query}

Answer:"""
    
#CALLS OPENROUTER MODEL   
    try:
        completion = client.chat.completions.create(
            model="google/gemini-2.5-flash", 
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0,  # For consistent, deterministic responses
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        return f"Error calling LLM: {str(e)}"