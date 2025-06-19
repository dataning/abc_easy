# ----------------------------------
# Configuration & Dependencies
# ----------------------------------
import uuid
import requests
import time
import datetime
from requests.auth import HTTPBasicAuth


# ----------------------------------
# API Configuration
# ----------------------------------
# Authentication credentials
USERNAME = "dshad"                # Aladdin username
PASSWORD = "dasdasd"              # Aladdin password
CLIENT_ENV = "dasd"               # Client environment (e.g., "dev" or "tst")
API_KEY = "dasd"                  # API key goes here


# ----------------------------------
# Helper Functions
# ----------------------------------
def generate_headers():
    """Generate required headers for API requests."""
    return {
        'VND.com.blackrock.API-Key': API_KEY,
        'VND.com.blackrock.Origin-Timestamp':
            str(datetime.datetime.utcnow().replace(microsecond=0).astimezone().isoformat()),
        'VND.com.blackrock.Request-ID': str(uuid.uuid1())
    }


# ----------------------------------
# API Request Configuration
# ----------------------------------
# Chat completion request payload
CHAT_COMPLETION_REQUEST = {
    "chatCompletionMessages": [
        {
            "prompt": "I want you to act as a tour guide\\n",
            "promptRole": "system"
        },
        {
            "prompt": "Where is London?",
            "promptRole": "user"
        }
    ],
    "modelId": "gpt-4o",
    "modelParam": {
        "openAI": {
            "frequencyPenalty": 0.12,
            "maxTokens": 500
        }
    }
}


# ----------------------------------
# API Endpoints
# ----------------------------------
COMPUTE_ASYNC_URL = f"https://{CLIENT_ENV}.blackrock.com/api/ai-platform/toolkit/chat-completion/v1/chatCompletions:compute"
LONG_RUNNING_URL = f"https://{CLIENT_ENV}.blackrock.com/api/ai-platform/toolkit/chat-completion/v1/longRunningOperations"


# ----------------------------------
# API Execution
# ----------------------------------
def execute_async_request():
    """Execute asynchronous chat completion request and wait for results."""
    # Start asynchronous request
    response_post = requests.post(
        COMPUTE_ASYNC_URL,
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        headers=generate_headers(),
        json=CHAT_COMPLETION_REQUEST
    )
    
    # Check if the POST request was successful
    if response_post.status_code != 200:
        raise Exception(f"POST request failed with status {response_post.status_code}: {response_post.text}")
    
    # Parse response and check for 'id' key
    try:
        post_response_data = response_post.json()
    except ValueError as e:
        raise Exception(f"Failed to parse POST response as JSON: {e}")
    
    if 'id' not in post_response_data:
        raise Exception(f"POST response missing 'id' field. Response: {post_response_data}")
    
    # Extract operation ID
    operation_id = post_response_data['id']
    operation_url = f"{LONG_RUNNING_URL}/{operation_id}"

    # Wait for results with retries
    max_retries = 10
    wait_time = 2

    for _ in range(max_retries):
        result_response = requests.get(
            operation_url,
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            headers=generate_headers()
        )
        
        # Check if GET request was successful
        if result_response.status_code != 200:
            print(f"Warning: GET request failed with status {result_response.status_code}")
            time.sleep(wait_time)
            continue
            
        try:
            result_data = result_response.json()
        except ValueError:
            print("Warning: Failed to parse GET response as JSON")
            time.sleep(wait_time)
            continue

        if result_data.get('done'):
            return result_data

        time.sleep(wait_time)

    raise Exception("Request timed out after maximum retries")


# ----------------------------------
# Main Execution
# ----------------------------------
if __name__ == "__main__":
    try:
        # Execute and get results
        result = execute_async_request()

        # Print basic operation info
        print("Operation ID:", result['id'])
        print("Status:", "Completed" if result['done'] else "Failed")

        # Check if response contains expected structure
        if result.get('response') and 'chatCompletion' in result['response']:
            chat_completion = result['response']['chatCompletion']

            # Print metadata if available
            # if 'chatCompletionMetadata' in chat_completion:
            #     metadata = chat_completion['chatCompletionMetadata']
            #     print("\nResponse Metadata:")
            #     print(f"  - Prompt Tokens: {metadata.get('promptTokenCount', 'N/A')}")
            #     print(f"  - Completion Tokens: {metadata.get('completionTokenCount', 'N/A')}")
            #     print(f"  - Total Tokens: {metadata.get('totalTokenCount', 'N/A')}")
            # else:
            #     print("\nWarning: chatCompletionMetadata not found in response")

            # Print content if available
            if 'chatCompletionContent' in chat_completion:
                print("\nGenerated JSON Response:")
                print(chat_completion['chatCompletionContent'])
            else:
                print("\nWarning: chatCompletionContent not found in response")
        else:
            print("\nError: Unexpected response structure")
            print("Raw response:", result)

    except Exception as e:
        print(f"\nError occurred: {e}")
        if 'result' in locals():
            print("Last known response:", result)