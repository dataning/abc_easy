"""
Script to find Polymarket event IDs by searching through market data.
"""
from get_data_clean import MarketData
import polars as pl

def find_event_by_keyword(keyword: str, limit: int = 10000):
    """
    Search for events containing a keyword and show their IDs.
    
    Args:
        keyword: Search term (e.g., "mayoral", "election", "Trump")
        limit: Number of markets to fetch from API
    
    Returns:
        DataFrame with matching events
    """
    print(f"Searching for '{keyword}' in Polymarket events...")
    print("(This may take a minute...)\n")
    
    # Fetch latest markets
    df = MarketData.fetch_latest_markets(limit=limit, active_only=False)
    
    # Search in event titles (case-insensitive)
    results = (
        df.filter(
            pl.col('event_title').str.to_lowercase().str.contains(keyword.lower())
        )
        .unique(subset=['event_id'])  # One row per unique event
        .select(['event_id', 'event_title', 'event_ticker', 'end_date', 'active'])
        .sort('end_date', descending=True)
    )
    
    if len(results) == 0:
        print(f"No events found containing '{keyword}'")
        return results
    
    print(f"Found {len(results)} events containing '{keyword}':\n")
    print("=" * 100)
    
    for row in results.iter_rows(named=True):
        active_status = "ACTIVE" if row['active'] else "CLOSED"
        print(f"ID: {row['event_id']:10s} | {active_status:8s} | {row['event_title']}")
        print(f"    End Date: {row['end_date']} | Ticker: {row['event_ticker']}")
        print("-" * 100)
    
    return results


def find_event_by_url_slug(slug: str):
    """
    Find event ID from URL slug (the text in the URL).
    
    Example:
        URL: https://polymarket.com/event/new-york-city-mayoral-election
        Slug: "new-york-city-mayoral-election"
    
    Args:
        slug: The URL slug from the Polymarket event page
    """
    print(f"Searching for event with slug: {slug}...")
    
    # Fetch markets and search by converting title to slug format
    df = MarketData.fetch_latest_markets(limit=10000, active_only=False)
    
    # Create a slug-like version of titles to match
    results = (
        df.filter(
            pl.col('event_title')
            .str.to_lowercase()
            .str.replace_all(r'[^\w\s-]', '')
            .str.replace_all(r'\s+', '-')
            .str.contains(slug.replace('-', '.*'))
        )
        .unique(subset=['event_id'])
        .select(['event_id', 'event_title', 'end_date'])
    )
    
    if len(results) > 0:
        print(f"\nFound {len(results)} matching event(s):\n")
        for row in results.iter_rows(named=True):
            print(f"Event ID: {row['event_id']}")
            print(f"Title: {row['event_title']}")
            print(f"End Date: {row['end_date']}")
            print("-" * 80)
        return results
    else:
        print("No matching event found. Try searching by keyword instead.")
        return pl.DataFrame()


def inspect_event(event_id: str):
    """
    Fetch and display detailed information about an event.
    
    Args:
        event_id: The event ID to inspect
    """
    print(f"Fetching event ID: {event_id}...\n")
    
    event = MarketData.fetch_event_by_id(event_id)
    
    if not event:
        print(f"Could not fetch event {event_id}")
        return
    
    print("=" * 80)
    print(f"Event ID: {event.get('id')}")
    print(f"Title: {event.get('title')}")
    print(f"Ticker: {event.get('ticker')}")
    print(f"End Date: {event.get('endDate')}")
    print(f"Active: {event.get('active')}")
    print(f"Closed: {event.get('closed')}")
    print("=" * 80)
    
    markets = event.get('markets', [])
    print(f"\nThis event has {len(markets)} markets:")
    print("-" * 80)
    
    for i, market in enumerate(markets, 1):
        print(f"{i}. {market.get('question')}")
        print(f"   Price: {market.get('lastTradePrice')}")
        print(f"   Volume: ${market.get('volume', 0):,.0f}")
        print(f"   Active: {market.get('active')}")
        print()


if __name__ == "__main__":
    import sys
    
    # Example 1: Find NYC mayoral election
    print("EXAMPLE 1: FINDING NYC MAYORAL ELECTION")
    print("=" * 100)
    results = find_event_by_keyword("new york city mayoral")
    
    # Example 2: Search for Trump-related events
    print("\n\nEXAMPLE 2: FINDING TRUMP EVENTS")
    print("=" * 100)
    trump_events = find_event_by_keyword("trump", limit=5000)
    
    # Example 3: Inspect a specific event
    if len(results) > 0:
        first_event_id = results.row(0, named=True)['event_id']
        print(f"\n\nEXAMPLE 3: INSPECTING EVENT {first_event_id}")
        print("=" * 100)
        inspect_event(first_event_id)
    
    # Interactive mode
    print("\n\n" + "=" * 100)
    print("INTERACTIVE MODE")
    print("=" * 100)
    print("You can use this script to find event IDs:")
    print("  python find_event_id.py")
    print("\nOr import and use the functions:")
    print("  from find_event_id import find_event_by_keyword, inspect_event")
    print("  results = find_event_by_keyword('election')")
    print("  inspect_event('23246')")
