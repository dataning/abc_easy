from openai import OpenAI
import json

# Initialize OpenAI client that points to the local LM Studio server
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# Example of streaming response
def stream_response(prompt: str):
    """Stream a response from the LLM"""
    response = client.chat.completions.create(
        model="gemma-3n-e4b-it-mlx",  # Replace with your actual model name
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # New line at the end

# Test streaming
stream_response("What is the meaning of life?")

# Example of basic completion
def get_completion(prompt: str, max_tokens: int = 100) -> str:
    """Get a simple completion from the LLM"""
    response = client.chat.completions.create(
        model="your-model",  # Replace with your actual model name
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

result = get_completion("My name is", max_tokens=100)
print(result)


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment using structured JSON output"""
    
    # Define the JSON schema for sentiment analysis
    sentiment_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative"]
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["sentiment", "confidence"]
            }
        }
    }
    
    # Create the conversation
    messages = [
        {
            "role": "system", 
            "content": "You are a sentiment analyzer. Analyze the given text and return the sentiment (positive, neutral, or negative) along with your confidence level (0.0 to 1.0)."
        },
        {
            "role": "user", 
            "content": f"Analyze the sentiment of this text: \"{text}\""
        }
    ]
    
    try:
        # Get structured response from AI
        response = client.chat.completions.create(
            model="your-model",  # Replace with your actual model name
            messages=messages,
            response_format=sentiment_schema,
            max_tokens=100
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {
            "sentiment": "neutral", 
            "confidence": 0.5,
            "error": str(e)
        }

def create_characters(num_characters: str = "1-3") -> dict:
    """Create fictional characters using structured output"""
    
    # Define the expected response structure
    character_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "characters",
            "schema": {
                "type": "object",
                "properties": {
                    "characters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "occupation": {"type": "string"},
                                "personality": {"type": "string"},
                                "background": {"type": "string"}
                            },
                            "required": ["name", "occupation", "personality", "background"]
                        },
                        "minItems": 1,
                    }
                },
                "required": ["characters"]
            },
        }
    }

    # Define the conversation with the AI
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"Create {num_characters} fictional characters"}
    ]

    try:
        # Get response from AI
        response = client.chat.completions.create(
            model="your-model",  # Replace with your actual model name
            messages=messages,
            response_format=character_schema,
        )

        # Parse and return the results
        results = json.loads(response.choices[0].message.content)
        return results
    
    except Exception as e:
        print(f"Error creating characters: {e}")
        return {"characters": [], "error": str(e)}

# Example usage and tests
if __name__ == "__main__":
    print("=== Testing Sentiment Analysis ===")
    text = "I love sunny days, but I hate the traffic."
    sent = analyze_sentiment(text)
    print(f"Text: {text}")
    print(f"Result: {sent}")
    print()
    
    print("=== Testing Character Creation ===")
    characters = create_characters("2")
    print(json.dumps(characters, indent=2))
    print()
    
    print("=== Testing Basic Completion ===")
    completion = get_completion("Tell me a fun fact about space")
    print(f"Fun fact: {completion}")


