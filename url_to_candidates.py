"""
Complete workflow: From Polymarket URL to candidate probabilities.
Uses cached data to avoid refetching.
Usage: python url_to_candidates.py <url>
"""
import sys
import re
import os
from datetime import datetime, timedelta
from get_data_clean import MarketData, MarketConfig, APIConfig
import polars as pl


def get_cached_markets(cache_file: str = "polymarket_cache.parquet", 
                       max_age_hours: int = 24,
                       fetch_limit: int = 50000) -> pl.DataFrame:
    """
    Get markets from cache if recent, otherwise fetch fresh data.
    
    Args:
        cache_file: Path to cache file
        max_age_hours: Maximum age of cache in hours before refetching
        fetch_limit: Number of markets to fetch if cache is stale
        
    Returns:
        DataFrame with market data
    """
    
    # Check if cache exists and is recent
    if os.path.exists(cache_file):
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        
        if cache_age < timedelta(hours=max_age_hours):
            print(f"✓ Using cached data from {cache_file}")
            print(f"  Cache age: {cache_age.seconds // 3600}h {(cache_age.seconds % 3600) // 60}m")
            df = pl.read_parquet(cache_file)
            print(f"  Markets in cache: {len(df):,}\n")
            return df
        else:
            print(f"⚠ Cache is {cache_age.days} days old, fetching fresh data...\n")
    else:
        print(f"No cache found, fetching data...\n")
    
    # Fetch fresh data
    df = MarketData.fetch_latest_markets(limit=fetch_limit, active_only=False)
    
    # Save to cache
    df.write_parquet(cache_file)
    print(f"\n✓ Cached {len(df):,} markets to {cache_file}\n")
    
    return df


def extract_slug_from_url(url: str) -> str:
    """Extract the event slug from a Polymarket URL."""
    match = re.search(r'polymarket\.com/event/([^?&#]+)', url)
    if match:
        return match.group(1)
    return None


def find_event_id_by_slug(slug: str, cached_df: pl.DataFrame = None) -> tuple[str, dict]:
    """
    Search for event ID by URL slug.
    Uses cached DataFrame if provided, otherwise fetches fresh data.
    Returns: (event_id, event_data)
    """
    
    # Known event IDs (manually curated for common events)
    KNOWN_EVENTS = {
        'new-york-city-mayoral-election': '23246',
        '2024-presidential-election-popular-vote-margin': '21274',
        'fed-decision-in-october': '1343',  # Example, may need updating
    }
    
    # Check if this is a known event
    if slug in KNOWN_EVENTS:
        event_id = KNOWN_EVENTS[slug]
        print(f"✓ Found in known events database!")
        print(f"  Event ID: {event_id}\n")
        event_data = MarketData.fetch_event_by_id(event_id)
        if event_data:
            print(f"  Title: {event_data.get('title')}\n")
            return event_id, event_data
    
    print(f"Searching for event with slug: '{slug}'...")
    
    # Use cached data if provided
    if cached_df is None:
        print("No cache provided, fetching data...\n")
        cached_df = get_cached_markets()
    else:
        print(f"Searching in {len(cached_df):,} cached markets...\n")
    
    # Create slug-like versions of titles
    df = cached_df.with_columns([
        pl.col('event_title')
        .str.to_lowercase()
        .str.replace_all(r'[^\w\s-]', '')
        .str.replace_all(r'\s+', '-')
        .str.replace_all(r'-+', '-')
        .alias('title_slug')
    ])
    
    # Search for matches
    results = df.filter(
        pl.col('title_slug').str.contains(slug.replace('-', '.*'))
    ).unique(subset=['event_id']).head(5)  # Get top 5 matches
    
    if len(results) > 0:
        print(f"Found {len(results)} potential match(es):\n")
        
        # Show all matches
        for idx, row_data in enumerate(results.iter_rows(named=True), 1):
            print(f"{idx}. {row_data['event_title']}")
            print(f"   Event ID: {row_data['event_id']}")
            print(f"   End Date: {row_data['end_date']}")
            print()
        
        # Use the first match
        first_match = results.row(0, named=True)
        event_id = first_match['event_id']
        
        print(f"Using first match: {first_match['event_title']}")
        print(f"Event ID: {event_id}\n")
        
        # Fetch full event data
        event_data = MarketData.fetch_event_by_id(event_id)
        return event_id, event_data
    
    print("✗ Event not found in cached markets.\n")
    return None, None


def get_candidates_from_url(url: str, use_cache: bool = True) -> pl.DataFrame:
    """
    Complete workflow: URL -> Event ID -> Candidate probabilities.
    Uses cached data by default for faster lookups.
    
    Args:
        url: Full Polymarket event URL
        use_cache: Whether to use cached data (default: True)
        
    Returns:
        DataFrame with candidate probabilities
    """
    print("=" * 80)
    print("POLYMARKET URL TO CANDIDATES")
    print("=" * 80)
    print(f"URL: {url}\n")
    
    # Step 1: Extract slug
    slug = extract_slug_from_url(url)
    if not slug:
        print("✗ Could not extract event slug from URL")
        return None
    
    print(f"Step 1: Extracted slug: '{slug}'\n")
    
    # Step 2: Load cache if enabled
    cached_df = None
    if use_cache:
        cached_df = get_cached_markets()
    
    # Step 3: Find event ID
    event_id, event_data = find_event_id_by_slug(slug, cached_df)
    
    if not event_id or not event_data:
        print("\n✗ Could not find event. Try:")
        print("  1. Check if the URL is correct")
        print("  2. Refresh cache: python url_to_candidates.py --refresh")
        print("  3. The event might be very old or not available")
        return None
    
    print(f"Step 2: Found Event ID: {event_id}\n")
    
    # Step 4: Parse candidates
    print("Step 3: Parsing candidates...\n")
    candidates_df = MarketData.parse_election_candidates(event_data)
    
    if candidates_df.is_empty():
        print("✗ No active candidates found in this event")
        return None
    
    # Step 5: Display results
    print("=" * 80)
    print(f"EVENT: {event_data.get('title')}")
    print(f"EVENT ID: {event_id}")
    print(f"END DATE: {event_data.get('endDate')}")
    print("=" * 80)
    print(f"\nCANDIDATE PROBABILITIES ({len(candidates_df)} candidates):\n")
    
    for row in candidates_df.iter_rows(named=True):
        prob_pct = row['current_price'] * 100
        bar_length = int(prob_pct / 2)  # Scale for display
        bar = '█' * bar_length
        print(f"{row['candidate']:30s} {prob_pct:6.2f}% {bar}")
    
    print("\n" + "=" * 80)
    print("QUICK START CODE:")
    print("=" * 80)
    print(f"""
# Fetch this event
event = MarketData.fetch_event_by_id("{event_id}")
candidates_df = MarketData.parse_election_candidates(event)

# Get price histories
market_config = MarketConfig()
api_config = APIConfig()
market_data = MarketData(pl.DataFrame(), market_config, api_config)
price_histories = market_data.fetch_candidate_price_histories(candidates_df)
price_histories.write_csv("candidates_price_history.csv")
""")
    
    return candidates_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check for special flags
        if sys.argv[1] == "--refresh":
            print("Refreshing cache...")
            get_cached_markets(fetch_limit=100000)
            print("\n✓ Cache refreshed!")
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("""
Polymarket URL to Candidates Converter

Usage:
  python url_to_candidates.py <url>              Get candidates from URL
  python url_to_candidates.py --refresh          Refresh cache with latest data
  python url_to_candidates.py --help             Show this help

Examples:
  python url_to_candidates.py "https://polymarket.com/event/new-york-city-mayoral-election"
  python url_to_candidates.py "https://polymarket.com/event/fed-decision-in-october"

Features:
  - Uses cached data (24hr) for fast lookups
  - Automatically fetches fresh data if cache is old
  - Shows all matching events if multiple found
  - Displays candidate probabilities with visual bars
  - Provides copy-paste code for price histories
""")
            sys.exit(0)
        
        # Normal URL processing
        url = sys.argv[1]
        candidates = get_candidates_from_url(url)
    else:
        # Example usage
        print("EXAMPLE: NYC MAYORAL ELECTION")
        print("=" * 80)
        
        url = "https://polymarket.com/event/new-york-city-mayoral-election"
        candidates = get_candidates_from_url(url)
        
        print("\n\nUSAGE:")
        print("=" * 80)
        print("Command line:")
        print("  python url_to_candidates.py <polymarket_url>")
        print("  python url_to_candidates.py --refresh    # Refresh cache")
        print("\nIn Python:")
        print("  from url_to_candidates import get_candidates_from_url")
        print("  candidates = get_candidates_from_url('https://polymarket.com/event/...')")
        print("\n✓ First run creates cache (~1 min), subsequent runs are instant!")
