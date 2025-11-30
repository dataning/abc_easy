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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize configurations
    market_config = MarketConfig()
    api_config = APIConfig()
    
    # Example 1: Fetch all latest markets and save snapshot
    # -------------------------------------------------------
    print("=" * 80)
    print("EXAMPLE 1: FETCHING LATEST POLYMARKET DATA")
    print("=" * 80)
    initial_df = MarketData.fetch_latest_markets(limit=5000, active_only=False)
    MarketData.save_latest_snapshot(initial_df)
    
    print(f"\nTotal markets fetched: {len(initial_df)}")
    
    # Example 2: Get specific election candidates
    # -------------------------------------------
    print("\n" + "=" * 80)
    print("EXAMPLE 2: FETCHING NYC MAYORAL ELECTION CANDIDATES")
    print("=" * 80)
    
    # Fetch NYC mayoral election by event ID
    nyc_event = MarketData.fetch_event_by_id("23246")
    
    if nyc_event:
        # Get current candidate standings
        candidates_df = MarketData.parse_election_candidates(nyc_event)
        
        print(f"\nEvent: {nyc_event.get('title')}")
        print(f"End Date: {nyc_event.get('endDate')}")
        print(f"\nCandidate Standings:")
        print("-" * 80)
        
        for row in candidates_df.iter_rows(named=True):
            print(f"{row['candidate']:25s}: {row['current_price']*100:6.2f}%")
        
        # Example 3: Fetch price histories for all candidates
        # ---------------------------------------------------
        print("\n" + "=" * 80)
        print("EXAMPLE 3: FETCHING CANDIDATE PRICE HISTORIES")
        print("=" * 80)
        
        # Create market data instance
        market_data = MarketData(pl.DataFrame(), market_config, api_config)
        
        # Fetch price histories (this will take time - 1 API call per candidate)
        price_histories = market_data.fetch_candidate_price_histories(candidates_df)
        
        if not price_histories.is_empty():
            # Save to CSV
            filename = "nyc_mayoral_price_history.csv"
            price_histories.write_csv(filename)
            print(f"\n✓ Saved {len(price_histories)} price records to {filename}")
            
            # Show summary
            print("\nLatest Prices Summary:")
            print("-" * 80)
            summary = (
                price_histories
                .group_by('candidate')
                .agg([
                    pl.col('timestamp').max().alias('latest_timestamp'),
                    pl.col('price').last().alias('latest_price')
                ])
                .sort('latest_price', descending=True)
            )
            print(summary)
    
    # Example 4: Search and filter markets
    # ------------------------------------
    print("\n" + "=" * 80)
    print("EXAMPLE 4: SEARCHING FOR SPECIFIC MARKETS")
    print("=" * 80)
    
    # Search for markets containing "election"
    election_markets = initial_df.filter(
        pl.col('event_title').str.to_lowercase().str.contains('election')
    )
    print(f"\nFound {len(election_markets)} markets containing 'election'")
    
    # Show top 5 by volume
    print("\nTop 5 election markets by volume:")
    print("-" * 80)
    top_elections = (
        election_markets
        .sort('volume', descending=True)
        .select(['event_title', 'volume', 'end_date'])
        .head(5)
    )
    print(top_elections)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


# 1. Fetch all markets
initial_df = MarketData.fetch_latest_markets(limit=5000, active_only=False)
MarketData.save_latest_snapshot(initial_df)

# 2. Get NYC mayoral election candidates
nyc_event = MarketData.fetch_event_by_id("23246")
candidates_df = MarketData.parse_election_candidates(nyc_event)

# 3. Fetch candidate price histories
market_data = MarketData(pl.DataFrame(), market_config, api_config)
price_histories = market_data.fetch_candidate_price_histories(candidates_df)
price_histories.write_csv("nyc_mayoral_price_history.csv")

# 4. Search for specific markets
election_markets = initial_df.filter(
    pl.col('event_title').str.to_lowercase().str.contains('election')
)