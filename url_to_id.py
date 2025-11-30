"""
Quick script to get event ID from a Polymarket URL.
"""
import requests
import re

def get_event_id_from_url(url: str) -> str:
    """
    Extract event ID by fetching the page and finding it in the HTML/API response.
    
    Args:
        url: Full Polymarket event URL
        
    Returns:
        Event ID as string
    """
    # Try to get the page and extract event ID from HTML
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        html = response.text
        
        # Look for event ID in various places in the HTML
        # Pattern 1: "eventId":"23246"
        match = re.search(r'"eventId":"(\d+)"', html)
        if match:
            return match.group(1)
        
        # Pattern 2: "id":"23246" near "event"
        match = re.search(r'"event"[^}]*"id":"(\d+)"', html)
        if match:
            return match.group(1)
        
        # Pattern 3: event/slug format -> try API directly with slug
        slug_match = re.search(r'polymarket\.com/event/([^?&#]+)', url)
        if slug_match:
            slug = slug_match.group(1)
            print(f"Extracted slug: {slug}")
            print("Searching through API for matching event...")
            
            # Fetch from API and search
            api_url = "https://gamma-api.polymarket.com/events"
            params = {"limit": 100, "offset": 0}
            
            for _ in range(10):  # Try first 1000 events
                api_response = requests.get(api_url, params=params, timeout=30)
                if api_response.status_code == 200:
                    events = api_response.json()
                    if not events:
                        break
                    
                    for event in events:
                        # Check if title matches slug
                        title_slug = event.get('title', '').lower().replace(' ', '-').replace(',', '')
                        if slug.lower() in title_slug or title_slug in slug.lower():
                            event_id = event.get('id')
                            print(f"✓ Found matching event: {event.get('title')}")
                            return event_id
                    
                    params['offset'] += 100
                else:
                    break
        
        print("Could not find event ID in page")
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Test with NYC mayoral election URL
    url = "https://polymarket.com/event/new-york-city-mayoral-election?tid=1761505368300"
    
    print(f"Finding event ID for URL:")
    print(f"  {url}\n")
    
    event_id = get_event_id_from_url(url)
    
    if event_id:
        print(f"\n{'='*60}")
        print(f"Event ID: {event_id}")
        print(f"{'='*60}")
        print(f"\nYou can now use:")
        print(f"  nyc_event = MarketData.fetch_event_by_id(\"{event_id}\")")
    else:
        print("\nCould not determine event ID from URL")
