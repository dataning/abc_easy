from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import polars as pl
import requests

@dataclass
class MarketConfig:
    essential_columns: List[str] = None
    
    def __post_init__(self):
        if self.essential_columns is None:
            self.essential_columns = [
                "event_id", "event_ticker", "event_title", "clob_token_ids",
                "volume", "liquidity", "last_trade_price", "best_ask", "best_bid",
                "volume_24h", "start_date", "created_at", "end_date", "updated_at",
                "active", "closed", "outcomes", "outcome_prices"
            ]

@dataclass
class APIConfig:
    url: str = "https://clob.polymarket.com/prices-history"
    headers: dict = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Origin": "https://polymarket.com",
                "Connection": "keep-alive"
            }

@dataclass
class MarketData:
    df: pl.DataFrame
    config: MarketConfig
    api_config: APIConfig
    
    @staticmethod
    def load_parquet(file_path: str) -> pl.DataFrame:
        return pl.read_parquet(file_path)
    
    def prepare_essential_data(self) -> pl.DataFrame:
        """Prepare DataFrame with essential columns."""
        return (
            self.df
            .select(self.config.essential_columns)
            .with_columns([
                pl.col(['start_date', 'end_date', 'created_at', 'updated_at'])
                .str.to_datetime(time_zone="UTC")
                .dt.date(),
                pl.col('clob_token_ids')
                .str.replace_all(r'[\[\]"]', '')
                .str.split(", ")
                .list.first()
                .alias('clob_token_ids')
            ])
            .filter(pl.col("end_date").is_not_null())
            .sort("end_date", descending=True)
        )
    
    def get_price_history(self, market_id: str, interval: str = "all", fidelity: str = '15') -> Optional[pl.DataFrame]:
        """Fetch price history for a given market ID."""
        querystring = {
            "interval": interval,
            "market": market_id,
            "fidelity": fidelity
        }
        
        response = requests.get(self.api_config.url, headers=self.api_config.headers, params=querystring)
        if response.status_code == 200:
            data = response.json()
            if data['history']:
                return pl.DataFrame({
                    'timestamp': pl.Series([h['t'] for h in data['history']]),
                    'price': pl.Series([h['p'] for h in data['history']])
                }).with_columns([
                    pl.col('timestamp')
                    .cast(pl.Int64)
                    .mul(1000)
                    .cast(pl.Datetime('ms'))
                ])
        return None

    def process_market_data(self, df_filtered: pl.DataFrame) -> pl.DataFrame:
        """Process market data and combine price histories."""
        dfs = []
        total_markets = len(df_filtered)
        print(f"Processing {total_markets} markets...")
        
        for idx, row in enumerate(df_filtered.iter_rows(named=True), 1):
            print(f"Fetching price history for market {idx}/{total_markets}: {row['event_title'][:50]}...")
            df_prices = self.get_price_history(row['clob_token_ids'])
            if df_prices is not None:
                df_with_id = df_prices.with_columns([
                    pl.lit(row['clob_token_ids']).alias('clob_token_ids')
                ])
                dfs.append(df_with_id)
        
        if not dfs:
            print("No price history data found.")
            return pl.DataFrame()
        
        print(f"Combining data from {len(dfs)} markets...")
        price_history_df = pl.concat(dfs)
        return (
            price_history_df.join(
                df_filtered,
                on='clob_token_ids',
                how='left'
            )
            .with_columns([
                pl.col('timestamp').alias('full_timestamp'),
                pl.col('outcome_prices')
                .str.replace_all(r'[\[\]"]', '')
                .str.split(',')
                .list.get(0)
                .alias('yes_quote'),
                pl.col('outcome_prices')
                .str.replace_all(r'[\[\]"]', '')
                .str.split(',')
                .list.get(1)
                .alias('no_quote')
            ])
            .select([
                'full_timestamp',
                'price',
                'yes_quote',
                'no_quote',
                'event_title',
                'event_ticker',
                'end_date'
            ])
            .sort(['full_timestamp', 'end_date'], descending=True)
        )
    
    def save_market_data(self, df: pl.DataFrame, event_ticker: str) -> None:
        """Save DataFrame with event ticker and current date in filename."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        clean_ticker = event_ticker.replace("/", "_").replace(" ", "_").lower()
        filename = f"{clean_ticker}_{current_date}.parquet"
        df.write_parquet(filename)
        print(f"Data saved to {filename}")

    @staticmethod
    def fetch_latest_markets(limit: int = 1000, active_only: bool = True) -> pl.DataFrame:
        """Fetch the latest market data from Polymarket API."""
        base_url = "https://gamma-api.polymarket.com/events"
        all_markets = []
        offset = 0
        
        print("Fetching latest markets from Polymarket...")
        
        while True:
            params = {
                "limit": min(100, limit - offset),
                "offset": offset
            }
            
            if active_only:
                params["active"] = "true"
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                events = response.json()
                
                if not events:
                    break
                
                # Extract markets from events
                for event in events:
                    for market in event.get("markets", []):
                        market_data = {
                            "event_id": event.get("id"),
                            "event_ticker": event.get("ticker"),
                            "event_title": event.get("title"),
                            "market_question": market.get("question"),
                            "market_slug": market.get("slug"),
                            "group_item_title": market.get("groupItemTitle"),
                            "clob_token_ids": market.get("clobTokenIds"),
                            "volume": market.get("volume"),
                            "liquidity": market.get("liquidity"),
                            "last_trade_price": market.get("lastTradePrice"),
                            "best_ask": market.get("bestAsk"),
                            "best_bid": market.get("bestBid"),
                            "volume_24h": market.get("volume24hr"),
                            "start_date": market.get("startDate"),
                            "created_at": market.get("createdAt"),
                            "end_date": market.get("endDate"),
                            "updated_at": market.get("updatedAt"),
                            "active": market.get("active"),
                            "closed": market.get("closed"),
                            "outcomes": market.get("outcomes"),
                            "outcome_prices": market.get("outcomePrices"),
                        }
                        all_markets.append(market_data)
                
                offset += len(events)
                print(f"Fetched {len(all_markets)} markets...")
                
                if offset >= limit or len(events) < 100:
                    break
                    
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        print(f"Total markets fetched: {len(all_markets)}")
        return pl.DataFrame(all_markets)
    
    @staticmethod
    def save_latest_snapshot(df: pl.DataFrame, filename: str = None) -> None:
        """Save the latest market snapshot."""
        if filename is None:
            current_date = datetime.now().strftime("%Y-%m-%d")
            filename = f"polymarket_data_{current_date}.parquet"
        
        df.write_parquet(filename)
        print(f"Latest snapshot saved to {filename}")
    
    @staticmethod
    def fetch_event_by_id(event_id: str) -> Optional[dict]:
        """Fetch a specific event by its ID from the API."""
        base_url = f"https://gamma-api.polymarket.com/events/{event_id}"
        
        try:
            response = requests.get(base_url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching event {event_id}: {e}")
            return None
    
    @staticmethod
    def parse_election_candidates(event_data: dict) -> pl.DataFrame:
        """Parse election event data to extract candidate information."""
        import json
        
        candidates = []
        for market in event_data.get('markets', []):
            if not market.get('active'):
                continue
            
            question = market.get('question', '')
            # Extract candidate name from question
            if "Will " in question and " win" in question:
                candidate = question.split("Will ")[1].split(" win")[0]
            else:
                candidate = question
            
            # Parse prices
            outcome_prices = market.get('outcomePrices')
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices.replace("'", '"'))
            
            yes_price = float(outcome_prices[0]) if outcome_prices and len(outcome_prices) > 0 else 0.0
            
            # Parse token IDs
            clob_ids = market.get('clobTokenIds', [])
            if isinstance(clob_ids, str):
                clob_ids = json.loads(clob_ids.replace("'", '"'))
            token_id = clob_ids[0] if clob_ids else None
            
            candidates.append({
                'candidate': candidate,
                'current_price': yes_price,
                'token_id': token_id,
                'question': question,
                'volume': market.get('volume', 0),
                'liquidity': market.get('liquidity', 0),
            })
        
        return pl.DataFrame(candidates).sort('current_price', descending=True)
    
    def fetch_candidate_price_histories(self, candidates_df: pl.DataFrame) -> pl.DataFrame:
        """Fetch price history for all candidates and combine into single DataFrame."""
        all_histories = []
        
        total = len(candidates_df)
        print(f"Fetching price history for {total} candidates...")
        
        for idx, row in enumerate(candidates_df.iter_rows(named=True), 1):
            candidate = row['candidate']
            token_id = row['token_id']
            
            print(f"[{idx}/{total}] Fetching {candidate}...")
            
            df_prices = self.get_price_history(token_id)
            if df_prices is not None and len(df_prices) > 0:
                df_with_candidate = df_prices.with_columns([
                    pl.lit(candidate).alias('candidate')
                ])
                all_histories.append(df_with_candidate)
        
        if not all_histories:
            print("No price history found.")
            return pl.DataFrame()
        
        print(f"Combining data from {len(all_histories)} candidates...")
        combined = pl.concat(all_histories)
        
        return combined.select([
            'timestamp',
            'candidate', 
            'price'
        ]).sort(['timestamp', 'candidate'], descending=[True, False])
        """Fetch a specific event by its slug from the URL."""
        base_url = f"https://gamma-api.polymarket.com/events/{event_slug}"
        
        try:
            response = requests.get(base_url, timeout=30)
            response.raise_for_status()
            event = response.json()
            
            markets = []
            for market in event.get("markets", []):
                market_data = {
                    "event_id": event.get("id"),
                    "event_ticker": event.get("ticker"),
                    "event_title": event.get("title"),
                    "clob_token_ids": market.get("clobTokenIds"),
                    "volume": market.get("volume"),
                    "liquidity": market.get("liquidity"),
                    "last_trade_price": market.get("lastTradePrice"),
                    "best_ask": market.get("bestAsk"),
                    "best_bid": market.get("bestBid"),
                    "volume_24h": market.get("volume24hr"),
                    "start_date": market.get("startDate"),
                    "created_at": market.get("createdAt"),
                    "end_date": market.get("endDate"),
                    "updated_at": market.get("updatedAt"),
                    "active": market.get("active"),
                    "closed": market.get("closed"),
                    "outcomes": market.get("outcomes"),
                    "outcome_prices": market.get("outcomePrices"),
                }
                markets.append(market_data)
            
            print(f"Fetched event: {event.get('title')} with {len(markets)} markets")
            return pl.DataFrame(markets)
            
        except Exception as e:
            print(f"Error fetching event {event_slug}: {e}")
            return None


# Initialize configurations
market_config = MarketConfig()
api_config = APIConfig()

# Fetch only active markets
initial_df = MarketData.fetch_latest_markets(limit=50000, active_only=True)
MarketData.save_latest_snapshot(initial_df)


nyc_mayoral_search = initial_df.filter(
    (pl.col('event_title').str.to_lowercase().str.contains('new york')) &
    (pl.col('event_title').str.to_lowercase().str.contains('mayoral|mayor'))
)
print(f"NYC mayoral markets found: {len(nyc_mayoral_search)}")

if len(nyc_mayoral_search) > 0:
    nyc_mayoral_search.select(['event_title', 'event_ticker', 'end_date', 'active', 'closed']).head(20)

nyc_2025_election = initial_df.filter(
    (pl.col('event_ticker') == 'new-york-city-mayoral-election') &
    (pl.col('end_date').str.contains('2025'))
)

print(f"NYC 2025 Mayoral markets: {len(nyc_2025_election)}")
nyc_2025_election.select(['event_title', 'outcomes', 'outcome_prices', 'last_trade_price', 'active', 'closed'])


# Get more details about each market - need to see the question/description
nyc_active = nyc_2025_election.filter(pl.col('active') == True)
print(f"Active NYC mayoral markets: {len(nyc_active)}")

# Create a MarketData instance and process these markets
market_data_nyc = MarketData(nyc_active, market_config, api_config)
df_essential_nyc = market_data_nyc.prepare_essential_data()

# Show the markets
df_essential_nyc.select(['event_title', 'clob_token_ids', 'outcomes', 'outcome_prices', 'end_date'])


# Process and get price history for NYC mayoral election
final_df_nyc = market_data_nyc.process_market_data(df_essential_nyc)

# Save the data
if not final_df_nyc.is_empty():
    market_data_nyc.save_market_data(final_df_nyc, 'new-york-city-mayoral-election')
    print(final_df_nyc.head(20))

nyc_active.select(['clob_token_ids', 'outcomes', 'outcome_prices', 'volume', 'liquidity']).head(15)


# Get all columns to see if there's a question/description field
print(nyc_active.columns)

# If there's more info in the raw data
initial_df.filter(
    (pl.col('event_ticker') == 'new-york-city-mayoral-election') &
    (pl.col('end_date').str.contains('2025'))
).head(5)





# Recreate market_data with fresh data
market_data = MarketData(initial_df, market_config, api_config)
df_essential = market_data.prepare_essential_data()

# Filter for NBA markets
df_filtered = df_essential.filter(
    pl.col("event_title").str.contains("NBA")
).head(10)

df_filtered.select(['event_title', 'end_date', 'active'])


df_2025 = initial_df.filter(
    pl.col('end_date').str.contains('2025')
)
print(f"Markets with 2025 dates: {len(df_2025)}")
df_2025.select(['event_title', 'end_date']).head(10)


# Search for NBA in 2025 markets
df_2025_nba = df_2025.filter(
    pl.col('event_title').str.to_lowercase().str.contains('nba')
)
print(f"NBA markets in 2025: {len(df_2025_nba)}")

if len(df_2025_nba) > 0:
    df_2025_nba.select(['event_title', 'end_date', 'active'])
else:
    print("No NBA markets found in 2025 data")
    
# Check what sports/topics are available in 2025
df_2025.select('event_title').head(20)


# Fetch more markets without active filter
initial_df_all = MarketData.fetch_latest_markets(limit=10000, active_only=False)
print(f"Total markets fetched: {len(initial_df_all)}")

# Search again
nyc_markets_all = initial_df_all.filter(
    pl.col('event_title').str.to_lowercase().str.contains('mayoral|new york')
)
print(f"NYC markets in full dataset: {len(nyc_markets_all)}")
nyc_markets_all.select(['event_title', 'event_ticker', 'end_date'])

mayoral_2025 = nyc_markets_all.filter(
    pl.col('event_ticker').str.contains('mayoral-election')
)
mayoral_2025.select(['event_title', 'event_ticker', 'end_date', 'active', 'closed'])


# Fetch NYC markets with candidate names
nyc_df = MarketData.fetch_latest_markets(limit=10000, active_only=False)

# Filter for NYC mayoral 2025
nyc_2025 = nyc_df.filter(
    (pl.col('event_ticker') == 'new-york-city-mayoral-election') &
    (pl.col('end_date').str.contains('2025'))
)

# Show candidates
nyc_2025.select(['market_question', 'group_item_title', 'outcome_prices', 'active', 'clob_token_ids']).head(20)


# Use the large dataset you already fetched
nyc_2025 = initial_df.filter(
    (pl.col('event_ticker') == 'new-york-city-mayoral-election') &
    (pl.col('end_date').str.contains('2025'))
)

# Check available columns
print(initial_df.columns)

# Show what fields exist
nyc_2025.head(5)

# See all columns in your existing data
print(initial_df.columns)

# See all columns
print(initial_df.columns)

# Get full view without truncation
nyc_sample = initial_df.filter(
    pl.col('event_ticker') == 'new-york-city-mayoral-election'
).head(3)

# Print each column separately to see what's available
for col in nyc_sample.columns:
    print(f"\n{col}:")
    print(nyc_sample[col])

response = requests.get("https://gamma-api.polymarket.com/events/23246")
nyc_event = response.json()

print(f"Found: {nyc_event['title']}")
print(f"End date: {nyc_event.get('endDate')}")
print(f"\nCandidates ({len(nyc_event['markets'])} markets):")
print("-" * 80)

for market in nyc_event['markets']:
    question = market.get('question', '')
    prices = market.get('outcomePrices', [])
    yes_price = prices[0] if prices else 'N/A'
    active = market.get('active', False)
    clob_id = market.get('clobTokenIds', [])
    if clob_id:
        clob_id = clob_id[0] if isinstance(clob_id, list) else clob_id
    
    if active:
        print(f"[{yes_price:>6}] {question}")
        print(f"         Token ID: {clob_id}")
        print()



import json

print("NYC 2025 Mayoral Election - Current Prices:")
print("=" * 80)

for market in nyc_event['markets']:
    if market.get('active'):
        question = market.get('question', '')
        candidate = question.split("Will ")[1].split(" win")[0] if "Will " in question else question
        
        # Parse prices (they're returned as string representation of list)
        outcome_prices = market.get('outcomePrices')
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices.replace("'", '"'))
        
        if outcome_prices and len(outcome_prices) > 0:
            yes_price = float(outcome_prices[0])
        else:
            yes_price = 0.0
        
        clob_ids = market.get('clobTokenIds', [])
        if isinstance(clob_ids, str):
            clob_ids = json.loads(clob_ids.replace("'", '"'))
        token_id = clob_ids[0] if clob_ids else "N/A"
        
        print(f"{candidate:25s}: {yes_price*100:6.2f}%  (Token: {token_id[:20]}...)")

print("\n" + "=" * 80)



# Create MarketData instance
market_data = MarketData(initial_df, market_config, api_config)
market_data

# Prepare essential data
df_essential = market_data.prepare_essential_data()
df_essential

# Filter for specific market
df_filtered = df_essential.filter(
    (pl.col("event_title").str.contains("NBA")) &
    (pl.col("active") == True)  # Only active markets
).head(10)  # Limit to 10 markets for faster testing
df_filtered

print(f"Number of active NBA markets: {len(df_filtered)}")
df_filtered.select(['event_title', 'end_date', 'active'])

# Process market data
final_df = market_data.process_market_data(df_filtered)
final_df

# Save data using event ticker
if not final_df.is_empty():
    event_ticker = final_df['event_ticker'][0]
    market_data.save_market_data(final_df, event_ticker)


# Option 2: Load from existing parquet file
# initial_df = MarketData.load_parquet("polymarket_data_2024-12-15.parquet")
