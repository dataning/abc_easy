
# === xAI Grok API Example ===
import os
from openai import OpenAI

def test_xai_grok():
    """Test xAI Grok API with proper error handling"""
    
    messages = [
        {
            "role": "system",
            "content": "You are a highly intelligent AI assistant.",
        },
        {
            "role": "user",
            "content": "Where is York?",
        },
    ]

    # Fix: Use the API key directly instead of os.getenv()
    xai_client = OpenAI(
        base_url="https://api.x.ai/v1",
        api_key="xai-api-key",
    )

    try:
        completion = xai_client.chat.completions.create(
            model="grok-3-mini",  # or "grok-3-mini-fast"
            reasoning_effort="high",
            messages=messages,
            temperature=0.7,
        )

        # Check if reasoning content exists (it may not be available)
        message = completion.choices[0].message
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print("Reasoning Content:")
            print(message.reasoning_content)
            print()
        else:
            print("No reasoning content available for this model/request")
            print()

        print("Final Response:")
        print(message.content)
        print()

        # Handle usage statistics safely
        if completion.usage:
            print(f"Completion tokens: {completion.usage.completion_tokens}")
            print(f"Prompt tokens: {completion.usage.prompt_tokens}")
            print(f"Total tokens: {completion.usage.total_tokens}")
            
            # Check for reasoning tokens (may not be available)
            if (hasattr(completion.usage, 'completion_tokens_details') and 
                completion.usage.completion_tokens_details and
                hasattr(completion.usage.completion_tokens_details, 'reasoning_tokens')):
                print(f"Reasoning tokens: {completion.usage.completion_tokens_details.reasoning_tokens}")
            else:
                print("Reasoning tokens: Not available")
        
        return completion
        
    except Exception as e:
        print(f"Error calling xAI API: {e}")
        return None


from openai import OpenAI
    
client = OpenAI(
  api_key="xai-api-key",
  base_url="https://api.x.ai/v1",
)

completion = client.chat.completions.create(
  model="grok-3-mini",  # or "grok-3-mini-fast"
  reasoning_effort="high",
  messages=[
    {"role": "user", "content": "Where is York?"}
  ]
)

response = completion.sample()
print(response.content)


import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer xai-api-key"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "Provide me a digest of world news in the last 24 hours."
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "return_citations": True
    },
    "model": "grok-3-latest"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())


import os
from openai import OpenAI

messages = [
    {
        "role": "system",
        "content": "You are a highly intelligent AI assistant.",
    },
    {
        "role": "user",
        "content": "Where is York?",
    },
]

client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key="xai-api-key",
)

completion = client.chat.completions.create(
    model="grok-3-mini", # or "grok-3-mini-fast"
    # reasoning_effort="high",
    messages=messages,
    temperature=0.7,
)

# print("Reasoning Content:")
# print(completion.choices[0].message.reasoning_content)

print("\nFinal Response:")
print(completion.choices[0].message.content)

print("\nNumber of completion tokens (input):")
print(completion.usage.completion_tokens)

print("\nNumber of reasoning tokens (input):")
print(completion.usage.completion_tokens_details.reasoning_tokens)



import os
from openai import OpenAI

XAI_API_KEY = "xai-api-key"
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

def stream_xai_response(prompt: str, system_prompt: str = "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."):
    """
    Stream a response from xAI Grok API with proper None handling
    """
    XAI_API_KEY = "xai-api-key"
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )
    
    try:
        stream = client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True
        )
        
        print(f"🤖 Grok: ", end="", flush=True)
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print("\n")  # New line at the end
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")

def quick_streaming_test():
    """Quick test of the fixed streaming function"""
    print("=== Testing Fixed Streaming ===")
    stream_xai_response("What is the meaning of life in 2 sentences?")

if __name__ == "__main__":
    # Uncomment this line to test the fixed streaming
    quick_streaming_test()
    # pass


import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer xai-api-key"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "Provide me a digest of world news in the last 24 hours."
        }
    ],
    "search_parameters": {
        "mode": "auto"
    },
    "model": "grok-3-mini"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())


import os
import requests
from openai import OpenAI
import json
from datetime import datetime, timedelta
from urllib.parse import urlparse
import time

XAI_API_KEY = "xai-api-key"

def grok_with_live_search(
    prompt: str,
    mode: str = "auto",
    sources: list = None,
    return_citations: bool = True,
    max_search_results: int = 20,
    from_date: str = None,
    to_date: str = None,
    country: str = None,
    stream: bool = False,
    timeout: int = 60,
    max_retries: int = 2
) -> dict:
    """
    Enhanced Grok API call with comprehensive Live Search capabilities and timeout handling
    
    Args:
        prompt: The user's question/request
        mode: "auto", "on", or "off" for search
        sources: List of data sources to use
        return_citations: Whether to return source citations
        max_search_results: Maximum number of search results to consider
        from_date: Start date for search (YYYY-MM-DD format)
        to_date: End date for search (YYYY-MM-DD format)
        country: ISO alpha-2 country code for geo-targeted search
        stream: Whether to stream the response
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
    
    Returns:
        API response as dict
    """
    
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }
    
    # Build search parameters
    search_params = {
        "mode": mode,
        "return_citations": return_citations,
        "max_search_results": max_search_results
    }
    
    # Add date range if specified
    if from_date:
        search_params["from_date"] = from_date
    if to_date:
        search_params["to_date"] = to_date
    
    # Handle sources configuration
    if sources is None:
        # Default sources (web and x)
        sources = [{"type": "web"}, {"type": "x"}]
    
    # Add country parameter to web/news sources if specified
    if country:
        for source in sources:
            if source["type"] in ["web", "news"]:
                source["country"] = country
    
    search_params["sources"] = sources
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "search_parameters": search_params,
        "model": "grok-3-mini",
        "stream": stream
    }
    
    # Retry logic
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"🔄 Retry attempt {attempt + 1}/{max_retries + 1}...")
            
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                stream=stream,
                timeout=timeout
            )
            
            if stream:
                return response  # Return the streaming response object
            else:
                # Check status code first
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('error', {}).get('message', f"HTTP {response.status_code}")
                    except:
                        error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    return {"error": f"API Error: {error_msg}"}
                
                # Parse response
                try:
                    response_data = response.json()
                    
                    # Validate response structure
                    if not isinstance(response_data, dict):
                        return {"error": f"Invalid response format: expected dict, got {type(response_data)}"}
                    
                    if 'choices' not in response_data:
                        return {"error": f"Missing 'choices' in response: {response_data}"}
                    
                    return response_data
                    
                except json.JSONDecodeError as e:
                    return {"error": f"JSON decode error: {e}. Response: {response.text[:200]}"}
        
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait_time = (attempt + 1) * 10  # Exponential backoff
                print(f"⏱️  Request timed out. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return {"error": "Request timed out after multiple attempts"}
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait_time = (attempt + 1) * 5
                print(f"🔗 Connection error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return {"error": f"Connection error: {str(e)}"}
        
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    return {"error": "Maximum retry attempts exceeded"}

def format_citations(citations: list, title: str = "Sources") -> None:
    """Format and display citations in a clean, organized way"""
    if not citations:
        return
        
    print(f"\n📚 {title} ({len(citations)} total):")
    print("=" * 60)
    
    # Group citations by domain for better organization
    citation_groups = {}
    for citation in citations:
        try:
            domain = urlparse(citation).netloc.replace('www.', '')
            if domain not in citation_groups:
                citation_groups[domain] = []
            citation_groups[domain].append(citation)
        except:
            if 'other' not in citation_groups:
                citation_groups['other'] = []
            citation_groups['other'].append(citation)
    
    # Display grouped citations
    for domain, urls in citation_groups.items():
        print(f"\n🌐 {domain.upper()}:")
        for i, url in enumerate(urls, 1):
            print(f"   [{i}] {url}")

def quick_news_digest():
    """Get a quick news digest from the last 24 hours"""
    print("=== 📰 Quick News Digest (Last 24 Hours) ===")
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    
    result = grok_with_live_search(
        prompt="Provide me a digest of the top 5 world news stories in the last 24 hours. Structure each story clearly with headlines and key details.",
        mode="on",
        sources=[{"type": "news"}, {"type": "web"}],
        from_date=yesterday,
        to_date=today,
        max_search_results=15,
        timeout=45
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"🤖 {result['choices'][0]['message']['content']}")
    
    # Enhanced citation formatting
    if 'citations' in result:
        format_citations(result['citations'], "News Sources")

def search_specific_topic(topic: str, days_back: int = 7):
    """Search for a specific topic in recent news"""
    print(f"=== 🔍 Searching for: {topic} (Last {days_back} days) ===")
    
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    result = grok_with_live_search(
        prompt=f"Find the latest news and updates about {topic}. Provide a comprehensive summary with key developments, trends, and important details.",
        mode="on",
        sources=[
            {"type": "news"},
            {"type": "web"},
            {"type": "x"}
        ],
        from_date=from_date,
        max_search_results=10,
        timeout=45
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"🤖 {result['choices'][0]['message']['content']}")
    
    if 'citations' in result:
        format_citations(result['citations'], f"Sources for {topic}")

def search_x_handles(handles: list, query: str):
    """Search specific X handles for information"""
    print(f"=== 🐦 Searching X handles: {', '.join(handles)} ===")
    
    result = grok_with_live_search(
        prompt=f"What are {', '.join(handles)} saying about {query}? Summarize their recent posts and key messages.",
        mode="on",
        sources=[{"type": "x", "x_handles": handles}],
        max_search_results=15,
        timeout=30
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"🤖 {result['choices'][0]['message']['content']}")
    
    if 'citations' in result:
        format_citations(result['citations'], f"X Posts from {', '.join(handles)}")

def search_specific_websites(websites: list, query: str):
    """Search only specific websites"""
    print(f"=== 🌐 Searching websites: {', '.join(websites)} ===")
    
    result = grok_with_live_search(
        prompt=f"Search for comprehensive information about {query} from these specific sources.",
        mode="on",
        sources=[{"type": "web", "allowed_websites": websites}],
        max_search_results=10,
        timeout=30
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"🤖 {result['choices'][0]['message']['content']}")
    
    if 'citations' in result:
        format_citations(result['citations'], f"Results from {', '.join(websites)}")

def stream_live_search_response(prompt: str):
    """Stream a live search response"""
    print("=== 🌊 Streaming Live Search Response ===")
    print(f"Query: {prompt}")
    print("🤖 Grok: ", end="", flush=True)
    
    response = grok_with_live_search(
        prompt=prompt,
        mode="auto",
        stream=True,
        timeout=60
    )
    
    try:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data != '[DONE]':
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content')
                                if content:
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
        print("\n")
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")

def financial_news_search(company: str, days_back: int = 3):
    """Search for financial news about a specific company"""
    print(f"=== 💰 Financial News: {company} (Last {days_back} days) ===")
    
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    result = grok_with_live_search(
        prompt=f"Find the latest financial news, earnings reports, and market analysis for {company}. Include stock performance and analyst opinions. Structure the response with clear sections for: 1) Latest News, 2) Earnings/Financial Performance, 3) Market Analysis, 4) Stock Performance, 5) Analyst Opinions.",
        mode="on",
        sources=[
            {"type": "news"},
            {"type": "web", "allowed_websites": ["bloomberg.com", "reuters.com", "cnbc.com", "marketwatch.com", "yahoo.com"]}
        ],
        from_date=from_date,
        max_search_results=12,
        timeout=45
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result
    
    print(f"🤖 {result['choices'][0]['message']['content']}")
    
    # Enhanced citation formatting for financial news
    if 'citations' in result:
        format_citations(result['citations'], f"Financial Sources for {company}")
    
    return result

def quick_financial_news(company: str, days_back: int = 3):
    """Quick financial news search with reduced complexity"""
    print(f"=== ⚡ Quick Financial News: {company} (Last {days_back} days) ===")
    
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    result = grok_with_live_search(
        prompt=f"Provide a concise summary of the latest financial news for {company}. Include: 1) Recent news highlights, 2) Stock performance, 3) Key analyst views. Keep it brief but informative.",
        mode="on",
        sources=[
            {"type": "news"},
            {"type": "web", "allowed_websites": ["yahoo.com", "cnbc.com", "reuters.com"]}
        ],
        from_date=from_date,
        max_search_results=6,
        timeout=30,
        max_retries=1
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        
        # Last resort - try without date restrictions
        print("🔄 Trying without date restrictions...")
        result = grok_with_live_search(
            prompt=f"What's the latest news about {company}? Provide a brief financial summary.",
            mode="on",
            sources=[{"type": "news"}],
            max_search_results=5,
            timeout=20
        )
        
        if "error" in result:
            print(f"❌ Final attempt failed: {result['error']}")
            return result
    
    print(f"🤖 {result['choices'][0]['message']['content']}")
    
    if 'citations' in result:
        format_citations(result['citations'], f"Quick Sources for {company}")
    
    return result

def financial_news_enhanced(company: str, days_back: int = 3):
    """Enhanced financial news search with better structure and numbered citations"""
    print(f"=== 💰 Enhanced Financial News: {company} (Last {days_back} days) ===")
    
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    result = grok_with_live_search(
        prompt=f"""Find comprehensive financial information for {company}. Structure your response with these sections:

## 1. Latest Financial News
[Recent developments, partnerships, announcements]

## 2. Earnings & Performance  
[Quarterly results, revenue, key financial metrics]

## 3. Market Analysis
[Industry position, competitive analysis, market outlook]

## 4. Stock Performance
[Recent price movements, trading patterns, market cap]

## 5. Analyst Opinions
[Ratings, price targets, recommendations]

Include specific data points and mention sources when available.""",
        mode="on",
        sources=[
            {"type": "news"},
            {"type": "web", "allowed_websites": ["bloomberg.com", "reuters.com", "cnbc.com", "marketwatch.com", "yahoo.com"]}
        ],
        from_date=from_date,
        max_search_results=10,
        timeout=45,
        max_retries=1
    )
    
    # Enhanced error checking
    if not isinstance(result, dict):
        print(f"❌ Error: Invalid response type: {type(result)}")
        return {"error": f"Invalid response type: {type(result)}"}
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result
    
    # Validate response structure
    if 'choices' not in result or len(result['choices']) == 0:
        print(f"❌ Error: No choices in response")
        return {"error": "No choices in response"}
    
    if 'message' not in result['choices'][0]:
        print(f"❌ Error: No message in response")
        return {"error": "No message in response"}
    
    # Extract content safely
    try:
        content = result['choices'][0]['message']['content']
        print(f"🤖 {content}")
    except (KeyError, IndexError, TypeError) as e:
        print(f"❌ Error extracting content: {e}")
        return {"error": f"Error extracting content: {e}"}
    
    # Enhanced citation formatting with numbered references
    citations = result.get('citations', [])
    if citations:
        print(f"\n📈 Financial Sources for {company} ({len(citations)} total):")
        print("=" * 70)
        
        for i, citation in enumerate(citations, 1):
            try:
                domain = urlparse(citation).netloc.replace('www.', '')
                print(f"[{i:2d}] {domain}")
                print(f"     {citation}")
            except Exception as e:
                print(f"[{i:2d}] {citation}")
            
            # Add spacing every 3 citations for readability
            if i % 3 == 0 and i < len(citations):
                print()
    else:
        print(f"\n📈 No citations available for {company}")
    
    return result

def financial_news_enhanced_with_fallback(company: str, days_back: int = 3):
    """Enhanced financial news with automatic fallback to simpler search if timeout occurs"""
    print(f"=== 💰 Enhanced Financial News: {company} (Last {days_back} days) ===")
    
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    # Try enhanced search first
    try:
        print("🔄 Attempting comprehensive financial analysis...")
        result = grok_with_live_search(
            prompt=f"""Find comprehensive financial information for {company}. Structure your response with these sections:

## 1. Latest Financial News
[Recent developments, partnerships, announcements]

## 2. Earnings & Performance  
[Quarterly results, revenue, key financial metrics]

## 3. Market Analysis
[Industry position, competitive analysis, market outlook]

## 4. Stock Performance
[Recent price movements, trading patterns, market cap]

## 5. Analyst Opinions
[Ratings, price targets, recommendations]

Include specific data points and mention sources when available.""",
            mode="on",
            sources=[
                {"type": "news"},
                {"type": "web", "allowed_websites": ["bloomberg.com", "reuters.com", "cnbc.com", "marketwatch.com", "yahoo.com"]}
            ],
            from_date=from_date,
            max_search_results=10,
            timeout=45,
            max_retries=1
        )
        
        # Enhanced error checking
        if not isinstance(result, dict):
            raise Exception(f"Invalid response type: {type(result)}")
        
        if "error" in result:
            if "timed out" in result["error"].lower():
                raise TimeoutError("Enhanced search timed out")
            else:
                print(f"❌ Enhanced search failed: {result['error']}")
                raise Exception("Enhanced search failed")
        
        # Validate response structure
        if 'choices' not in result or len(result['choices']) == 0:
            raise Exception("No choices in response")
        
        if 'message' not in result['choices'][0]:
            raise Exception("No message in response")
        
        # Success - display results
        print("✅ Enhanced search completed successfully!")
        
        try:
            content = result['choices'][0]['message']['content']
            print(f"🤖 {content}")
        except (KeyError, IndexError, TypeError) as e:
            raise Exception(f"Error extracting content: {e}")
        
        # Enhanced citation formatting
        citations = result.get('citations', [])
        if citations:
            print(f"\n📈 Financial Sources for {company} ({len(citations)} total):")
            print("=" * 70)
            
            for i, citation in enumerate(citations, 1):
                try:
                    domain = urlparse(citation).netloc.replace('www.', '')
                    print(f"[{i:2d}] {domain}")
                    print(f"     {citation}")
                except:
                    print(f"[{i:2d}] {citation}")
                
                if i % 3 == 0 and i < len(citations):
                    print()
        
        return result
        
    except (TimeoutError, Exception) as e:
        print(f"⚠️  Enhanced search failed ({str(e)})")
        print("🔄 Falling back to quick financial news...")
        
        # Fallback to quick search
        return quick_financial_news(company, days_back)
    
def enhanced_news_digest_with_citations():
    """Get a news digest with properly formatted citations"""
    print("=== 📰 Enhanced World News Digest (Last 24 Hours) ===")
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    
    result = grok_with_live_search(
        prompt="""Provide the top 5 most important world news stories from the last 24 hours. 
        Format each story as:
        
        ## Story [Number]: [Clear Headline]
        **Key Points:**
        - [Main development]
        - [Impact/significance]
        - [Current status]
        
        Include specific details, quotes from officials, and mention reliable sources when available.
        Cover a mix of international politics, economics, technology, and major social developments.""",
        mode="on",
        sources=[{"type": "news"}, {"type": "web"}],
        from_date=yesterday,
        to_date=today,
        max_search_results=20,
        timeout=50
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    content = result['choices'][0]['message']['content']
    citations = result.get('citations', [])
    
    print(f"🤖 {content}")
    
    # Professional news source formatting
    if citations:
        format_citations(citations, "Global News Sources")

def search_with_inline_citations(topic: str, days_back: int = 7):
    """Search with citations integrated into the response"""
    print(f"=== 🔍 Deep Dive Research: {topic} (Last {days_back} days) ===")
    
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    result = grok_with_live_search(
        prompt=f"""Research {topic} comprehensively from the last {days_back} days. 
        
        Structure your response with clear sections and include source attribution where possible:
        - When mentioning specific facts: "According to [Source/Publication], [fact]..."
        - When referencing data: "[Data point] (reported by [Source])"
        - When quoting: "[Quote]" - [Source/Person, Title]
        
        Cover:
        1. Recent Developments
        2. Key Trends & Patterns  
        3. Expert Analysis & Opinions
        4. Future Implications
        5. Important Statistics/Data
        
        Provide a comprehensive analysis with proper context and attribution.""",
        mode="on",
        sources=[
            {"type": "news"},
            {"type": "web"},
            {"type": "x"}
        ],
        from_date=from_date,
        max_search_results=15,
        timeout=50
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    content = result['choices'][0]['message']['content']
    citations = result.get('citations', [])
    
    print(f"🤖 {content}")
    
    if citations:
        print(f"\n🔗 All Research Sources for {topic}:")
        print("-" * 50)
        for i, citation in enumerate(citations, 1):
            try:
                domain = urlparse(citation).netloc.replace('www.', '')
                print(f"{i:2d}. [{domain}] {citation}")
            except:
                print(f"{i:2d}. {citation}")

def financial_news_comprehensive(company: str, days_back: int = 3):
    """Get comprehensive financial news by running multiple targeted searches"""
    print(f"=== 💰 Comprehensive Financial Analysis: {company} (Last {days_back} days) ===")
    
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    # Search 1: Major financial news sites
    print("\n🔍 Searching Major Financial News Sites...")
    result1 = grok_with_live_search(
        prompt=f"Find the latest financial news and market analysis for {company} from major financial publications.",
        mode="on",
        sources=[
            {"type": "news"},
            {"type": "web", "allowed_websites": ["bloomberg.com", "reuters.com", "cnbc.com", "marketwatch.com", "yahoo.com"]}
        ],
        from_date=from_date,
        max_search_results=8,
        timeout=40
    )
    
    # Search 2: Investment analysis sites
    print("\n🔍 Searching Investment Analysis Sites...")
    result2 = grok_with_live_search(
        prompt=f"Find analyst opinions, ratings, and investment analysis for {company}.",
        mode="on",
        sources=[
            {"type": "web", "allowed_websites": ["zacks.com", "seekingalpha.com", "morningstar.com", "tradingview.com", "fool.com"]}
        ],
        from_date=from_date,
        max_search_results=6,
        timeout=40
    )
    
    # Display results
    if "error" not in result1:
        print(f"\n📰 Major Financial News:")
        print("=" * 50)
        print(f"🤖 {result1['choices'][0]['message']['content']}")
        
        if 'citations' in result1:
            format_citations(result1['citations'], "Major Financial News Sources")
    else:
        print(f"\n❌ Major financial news search failed: {result1['error']}")
    
    if "error" not in result2:
        print(f"\n📊 Investment Analysis:")
        print("=" * 50)
        print(f"🤖 {result2['choices'][0]['message']['content']}")
        
        if 'citations' in result2:
            format_citations(result2['citations'], "Investment Analysis Sources")
    else:
        print(f"\n❌ Investment analysis search failed: {result2['error']}")
    
    return result1, result2

# Enhanced test functions
def test_live_search_examples():
    """Test various live search capabilities"""
    
    print("🚀 Testing Grok-3-Mini Live Search Capabilities\n")
    
    # 1. Enhanced news digest
    enhanced_news_digest_with_citations()
    print("\n" + "="*70 + "\n")
    
    # 2. Deep topic search
    search_with_inline_citations("artificial intelligence", days_back=5)
    print("\n" + "="*70 + "\n")
    
    # 3. Enhanced financial news with fallback
    financial_news_enhanced_with_fallback("Apple", days_back=3)
    print("\n" + "="*70 + "\n")
    
    # 4. Search specific X handles
    search_x_handles(["elonmusk", "grok"], "AI developments")
    print("\n" + "="*70 + "\n")
    
    # 5. Search specific websites
    search_specific_websites(["techcrunch.com", "theverge.com"], "latest AI news")
    print("\n" + "="*70 + "\n")

def enhanced_test_menu():
    """Enhanced test menu with all citation options and timeout handling"""
    print("\n🚀 Grok Live Search with Enhanced Citations & Timeout Handling")
    print("=" * 65)
    print("Choose a test to run:")
    print("1. Quick news digest")
    print("2. Search specific topic")
    print("3. Financial news (basic)")
    print("4. Search X handles")
    print("5. Stream live search")
    print("6. Full live search demo")
    print("7. ⚡ Quick financial news")
    print("8. 🆕 Enhanced financial news with citations")
    print("9. 🆕 Enhanced world news digest")
    print("10. 🆕 Deep research with inline citations")
    print("11. 🆕 Comprehensive financial analysis")
    print("12. 🆕 Enhanced demo with all new features")
    print("13. 🛡️ Enhanced financial with auto-fallback (recommended)")
    print("14. 🔧 Test simple API call (debug)")
    
    choice = input("\nEnter choice (1-14) or press Enter for enhanced demo: ").strip()
    
    if choice == "1":
        quick_news_digest()
    elif choice == "2":
        topic = input("Enter topic to search: ").strip() or "AI news"
        search_specific_topic(topic)
    elif choice == "3":
        company = input("Enter company name: ").strip() or "Microsoft"
        financial_news_search(company)
    elif choice == "4":
        handles = input("Enter X handles (comma-separated): ").strip().split(",") or ["grok"]
        query = input("Enter search query: ").strip() or "latest updates"
        search_x_handles([h.strip() for h in handles], query)
    elif choice == "5":
        prompt = input("Enter your question: ").strip() or "What's happening in tech today?"
        stream_live_search_response(prompt)
    elif choice == "6":
        test_live_search_examples()
    elif choice == "7":
        company = input("Enter company name: ").strip() or "BlackRock"
        quick_financial_news(company)
    elif choice == "8":
        company = input("Enter company name: ").strip() or "BlackRock"
        financial_news_enhanced(company)
    elif choice == "9":
        enhanced_news_digest_with_citations()
    elif choice == "10":
        topic = input("Enter topic to search: ").strip() or "artificial intelligence"
        days = int(input("Days back (default 7): ").strip() or "7")
        search_with_inline_citations(topic, days)
    elif choice == "11":
        company = input("Enter company name: ").strip() or "BlackRock"
        financial_news_comprehensive(company)
    elif choice == "12":
        print("🚀 Running enhanced demo with all new features...")
        test_live_search_examples()
    elif choice == "13":
        company = input("Enter company name: ").strip() or "BlackRock"
        financial_news_enhanced_with_fallback(company)
    else:
        print("Running enhanced demo with citations and fallback...")
        enhanced_news_digest_with_citations()
        print("\n" + "="*70 + "\n")
        financial_news_enhanced_with_fallback("Apple")

if __name__ == "__main__":
    enhanced_test_menu()