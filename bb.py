"""
A lightweight module for interacting with abc's AI Platform Chat Completion API.
"""

import uuid
import requests
import time
import datetime
from typing import Dict, List, Optional, Any
from requests.auth import HTTPBasicAuth


class abcAIClient:
    """abc AI Platform API Client"""
    
    def __init__(self, username: str, password: str, client_env: str, api_key: str):
        """
        Initialize the abc AI client.
        
        Args:
            username: Aladdin username
            password: Aladdin password
            client_env: Client environment (e.g., "dev", "tst")
            api_key: API key for authentication
        """
        self.username = username
        self.password = password
        self.client_env = client_env
        self.api_key = api_key
        
        # API endpoints
        self.compute_url = f"https://{client_env}"
        self.operations_url = f"https://{client_env}"
    
    def _generate_headers(self) -> Dict[str, str]:
        """Generate required headers for API requests."""
        return {
            'VND.com.abc.API-Key': self.api_key,
            'VND.com.abc.Origin-Timestamp': 
                str(datetime.datetime.utcnow().replace(microsecond=0).astimezone().isoformat()),
            'VND.com.abc.Request-ID': str(uuid.uuid1())
        }
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model_id: str = "gpt-4o",
        max_tokens: int = 500,
        frequency_penalty: float = 0.12,
        max_retries: int = 10,
        wait_time: int = 2
    ) -> Dict[str, Any]:
        """
        Execute chat completion request.
        
        Args:
            messages: List of message dicts with 'prompt' and 'promptRole' keys
            model_id: Model identifier (default: "gpt-4o")
            max_tokens: Maximum tokens in response
            frequency_penalty: Frequency penalty parameter
            max_retries: Maximum retry attempts
            wait_time: Wait time between retries (seconds)
            
        Returns:
            Complete API response dict
            
        Raises:
            Exception: If request fails or times out
        """
        # Build request payload
        payload = {
            "chatCompletionMessages": messages,
            "modelId": model_id,
            "modelParam": {
                "openAI": {
                    "frequencyPenalty": frequency_penalty,
                    "maxTokens": max_tokens
                }
            }
        }
        
        # Start async request
        response = requests.post(
            self.compute_url,
            auth=HTTPBasicAuth(self.username, self.password),
            headers=self._generate_headers(),
            json=payload
        )
        response.raise_for_status()
        
        # Get operation ID and poll for results
        operation_id = response.json()['id']
        operation_url = f"{self.operations_url}/{operation_id}"
        
        for _ in range(max_retries):
            result = requests.get(
                operation_url,
                auth=HTTPBasicAuth(self.username, self.password),
                headers=self._generate_headers()
            )
            result.raise_for_status()
            result_data = result.json()
            
            if result_data.get('done'):
                return result_data
            
            time.sleep(wait_time)
        
        raise Exception(f"Request timed out after {max_retries} retries")
    
    def simple_chat(self, user_prompt: str, system_prompt: str = "", **kwargs) -> str:
        """
        Simplified chat interface that returns just the content.
        
        Args:
            user_prompt: User's message
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for chat_completion
            
        Returns:
            Generated text content
        """
        messages = []
        if system_prompt:
            messages.append({"prompt": system_prompt, "promptRole": "system"})
        messages.append({"prompt": user_prompt, "promptRole": "user"})
        
        result = self.chat_completion(messages, **kwargs)
        
        # Extract content from response
        try:
            return result['response']['chatCompletion']['chatCompletionContent']
        except KeyError as e:
            raise Exception(f"Unexpected response structure: missing {e}")


# Convenience factory function
def create_client(username: str, password: str, client_env: str, api_key: str) -> abcAIClient:
    """Create a abc AI client with credentials."""
    return abcAIClient(username, password, client_env, api_key)


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        "username": "dshad",
        "password": "dasdasd", 
        "client_env": "dasd",
        "api_key": "dasd"
    }
    
    # Create client
    client = create_client(**config)
    
    try:
        # Simple usage
        response = client.simple_chat(
            user_prompt="Where is London?",
            system_prompt="I want you to act as a tour guide"
        )
        print("Response:", response)
        
        # Advanced usage with custom parameters
        messages = [
            {"prompt": "You are a helpful assistant", "promptRole": "system"},
            {"prompt": "Explain quantum computing in 2 sentences", "promptRole": "user"}
        ]
        
        full_response = client.chat_completion(
            messages=messages,
            max_tokens=100,
            frequency_penalty=0.0
        )
        
        print("\nFull response available in:", full_response.keys())
        
    except Exception as e:
        print(f"Error: {e}")

    
# from abc_api import create_client

# client = create_client("user", "pass", "env", "key")

# # Simple usage
# response = client.simple_chat("What is AI?")

# # Advanced usage  
# result = client.chat_completion(messages, max_tokens=1000)
