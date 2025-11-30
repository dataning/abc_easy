# Polymarket Data Analysis Toolkit

Complete toolkit for fetching, analyzing, and tracking prediction markets from Polymarket.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
uv pip install polars requests pandas google-api-python-client youtube-transcript-api

# Or using requirements
uv pip install -r requirements.txt
```

### 2. Get Candidate Probabilities from URL

```bash
# Just paste any Polymarket event URL
python url_to_candidates.py "https://polymarket.com/event/new-york-city-mayoral-election"
```

**First run**: Builds cache (~1 minute, fetches 131k markets)  
**Subsequent runs**: Instant lookups from cache!

### 3. Get Price History

Use the code from Step 2 output:

```python
from get_data_clean import MarketData, MarketConfig, APIConfig
import polars as pl

# Fetch event and candidates
event = MarketData.fetch_event_by_id('23246')
candidates_df = MarketData.parse_election_candidates(event)

# Get price histories
market_config = MarketConfig()
api_config = APIConfig()
market_data = MarketData(pl.DataFrame(), market_config, api_config)
price_histories = market_data.fetch_candidate_price_histories(candidates_df)

# Save to CSV
price_histories.write_csv('nyc_mayoral_price_history.csv')
```

## 📚 Core Scripts

### `url_to_candidates.py` - URL to Candidates Converter

**Get candidates from any Polymarket URL:**

```bash
# Basic usage
python url_to_candidates.py "https://polymarket.com/event/your-event-url"

# Refresh cache (once per day or when you need latest data)
python url_to_candidates.py --refresh

# Show help
python url_to_candidates.py --help
```

**Output:**
- Event ID
- All candidates with current probabilities
- Visual probability bars
- Copy-paste code for price histories

**Features:**
- ✅ Uses 24-hour cache for instant lookups
- ✅ Searches 131k+ markets
- ✅ Known events database for common markets
- ✅ Automatic cache refresh when stale

---

### `get_data_clean.py` - Complete Data Fetching

**Run full example workflow:**

```bash
python get_data_clean.py
```

**What it does:**
1. Fetches 5,000 latest markets
2. Saves snapshot: `polymarket_data_YYYY-MM-DD.parquet`
3. Gets NYC mayoral candidates (event 23246)
4. Fetches price histories for all candidates
5. Saves: `nyc_mayoral_price_history.csv`

**Python Usage:**

```python
from get_data_clean import MarketData, MarketConfig, APIConfig

# 1. Fetch latest markets
df = MarketData.fetch_latest_markets(limit=10000, active_only=False)

# 2. Search for specific markets
election_markets = df.filter(
    pl.col('event_title').str.contains('(?i)election')
)

# 3. Get specific event
event = MarketData.fetch_event_by_id("23246")
candidates = MarketData.parse_election_candidates(event)

# 4. Get price history for a token
market_data = MarketData(df, MarketConfig(), APIConfig())
price_history = market_data.get_price_history(token_id="your_token_id")
```

---

### `find_event_id.py` - Find Event IDs by Keyword

**Search for events:**

```bash
python find_event_id.py
```

**Python Usage:**

```python
from find_event_id import find_event_by_keyword, inspect_event

# Find by keyword
results = find_event_by_keyword("election")
results = find_event_by_keyword("trump")

# Inspect specific event
inspect_event("23246")
```

---

### `parse_transcript.py` - YouTube Transcript Parser

**Parse YouTube transcripts into interview format:**

```bash
python parse_transcript.py
```

**Configuration:**

Edit the `INPUT_FILE` variable in the script:

```python
INPUT_FILE = "transcripts/K4KMyLTPcgU_timestamped.txt"
```

**Output:**
- `K4KMyLTPcgU_interview.txt` - Clean interview format with speaker labels

---

### `youtube_down.py` - YouTube Transcript Downloader

**Download transcripts:**

```python
# Edit VIDEO_URL in the script
VIDEO_URL = "https://www.youtube.com/watch?v=K4KMyLTPcgU"

# Run
python youtube_down.py
```

**Output files:**
- `transcripts/VIDEO_ID.txt` - Plain text
- `transcripts/VIDEO_ID_timestamped.txt` - With timestamps

**Note:** May require VPN or cookies if rate-limited by YouTube.

---

## 🗂️ File Structure

```
Polymarket/
├── url_to_candidates.py          # URL → Candidates (with cache)
├── get_data_clean.py             # Complete data fetching toolkit
├── find_event_id.py              # Search events by keyword
├── parse_transcript.py           # YouTube transcript parser
├── youtube_down.py               # YouTube transcript downloader
├── polymarket_cache.parquet      # 24hr cache (auto-created)
├── polymarket_data_YYYY-MM-DD.parquet  # Daily snapshots
├── nyc_mayoral_price_history.csv # Example output
└── transcripts/                  # YouTube transcripts
    ├── VIDEO_ID.txt
    ├── VIDEO_ID_timestamped.txt
    └── VIDEO_ID_interview.txt
```

## 💡 Common Workflows

### Workflow 1: Track Election Candidates Daily

```bash
# Morning: Get latest standings
python url_to_candidates.py "https://polymarket.com/event/new-york-city-mayoral-election"

# Copy the event ID and get price history
python -c "
from get_data_clean import MarketData, MarketConfig, APIConfig
import polars as pl

event = MarketData.fetch_event_by_id('23246')
candidates = MarketData.parse_election_candidates(event)
market_data = MarketData(pl.DataFrame(), MarketConfig(), APIConfig())
price_histories = market_data.fetch_candidate_price_histories(candidates)
price_histories.write_csv('daily_snapshot.csv')
"
```

### Workflow 2: Compare Multiple Events

```bash
# Find related events
python find_event_id.py  # Search for "election"

# Get data for each event
python url_to_candidates.py "URL1"
python url_to_candidates.py "URL2"
```

### Workflow 3: Historical Analysis

```python
from get_data_clean import MarketData
import polars as pl

# Load old snapshot
old_df = pl.read_parquet("polymarket_data_2025-10-20.parquet")

# Compare with current
new_df = MarketData.fetch_latest_markets(limit=1000)

# Analyze changes
combined = old_df.join(new_df, on="event_id", suffix="_new")
price_changes = combined.with_columns([
    (pl.col("last_trade_price_new") - pl.col("last_trade_price")).alias("price_change")
])
```

## 🔧 Advanced Usage

### Custom Cache Settings

```python
from url_to_candidates import get_cached_markets

# Custom cache location and age
df = get_cached_markets(
    cache_file="my_cache.parquet",
    max_age_hours=12,  # Refresh every 12 hours
    fetch_limit=100000  # Fetch more markets
)
```

### Fetch Specific Date Range

```python
from get_data_clean import MarketData
import polars as pl

# Fetch markets
df = MarketData.fetch_latest_markets(limit=50000)

# Filter by date
filtered = df.filter(
    (pl.col('end_date') >= '2025-11-01') &
    (pl.col('end_date') <= '2025-12-31')
)
```

### Custom Price History Intervals

```python
# Get hourly data instead of 15-minute
price_history = market_data.get_price_history(
    market_id="token_id",
    interval="all",
    fidelity="60"  # 60-minute intervals
)
```

## 📊 Data Schema

### Market Data Columns

```
event_id             str     Event identifier
event_title          str     Human-readable event name
event_ticker         str     Short ticker symbol
clob_token_ids       str     Token IDs for price history
volume              float    Total trading volume
liquidity           float    Current liquidity
last_trade_price    float    Most recent trade price (0-1)
best_ask            float    Best ask price
best_bid            float    Best bid price
volume_24h          float    24-hour trading volume
start_date          date     Event start date
end_date            date     Event end date (expiry)
created_at          date     Creation timestamp
updated_at          date     Last update timestamp
active              bool     Is market active?
closed              bool     Is market closed?
outcomes            list     Possible outcomes
outcome_prices      list     Current prices for each outcome
```

### Price History Format

```
timestamp           datetime  Time of price point
candidate           str       Candidate/option name
price              float     Price (0-1, where 1 = 100%)
```

## 🐛 Troubleshooting

### Cache Issues

```bash
# Delete and rebuild cache
rm polymarket_cache.parquet
python url_to_candidates.py --refresh
```

### Event Not Found

1. Check URL is correct
2. Event might be very old (not in recent 131k markets)
3. Try direct event ID if known:
   ```python
   event = MarketData.fetch_event_by_id("23246")
   ```

### YouTube Rate Limiting

If transcript download fails:
1. Wait 30-60 minutes
2. Use different network/VPN
3. Export browser cookies to `cookies.txt`

### API Rate Limits

Polymarket API is generally permissive, but if you hit limits:
- Add delays between requests: `time.sleep(1)`
- Reduce `fetch_limit` parameter
- Use cached data instead of fresh fetches

## 📖 API Reference

### MarketData Class

**Static Methods:**

- `fetch_latest_markets(limit, active_only)` - Fetch markets from API
- `fetch_event_by_id(event_id)` - Get specific event
- `parse_election_candidates(event_data)` - Extract candidates from event
- `save_latest_snapshot(df, filename)` - Save data to parquet
- `load_parquet(file_path)` - Load saved data

**Instance Methods:**

- `prepare_essential_data()` - Clean and filter DataFrame
- `get_price_history(market_id, interval, fidelity)` - Get historical prices
- `fetch_candidate_price_histories(candidates_df)` - Get all candidate histories

### Helper Functions

**url_to_candidates.py:**

- `get_candidates_from_url(url, use_cache)` - Main workflow
- `get_cached_markets(cache_file, max_age_hours, fetch_limit)` - Cache management
- `extract_slug_from_url(url)` - Parse URL slug
- `find_event_id_by_slug(slug, cached_df)` - Find event ID

**find_event_id.py:**

- `find_event_by_keyword(keyword, limit)` - Search by keyword
- `inspect_event(event_id)` - Display event details

## 🎯 Known Event IDs

```python
KNOWN_EVENTS = {
    'new-york-city-mayoral-election': '23246',
    'fed-decision-in-october': '27824',
    # Add more as you discover them
}
```

Add to `url_to_candidates.py` line ~25 for instant lookups.

## 📝 Tips & Best Practices

1. **Cache everything**: Use `polymarket_cache.parquet` for fast lookups
2. **Daily snapshots**: Save daily parquet files for historical analysis
3. **Rate limiting**: Add 1-2 second delays for large batches
4. **Event IDs**: Keep a list of important event IDs for quick access
5. **Price histories**: Fetch once and save - they're expensive API calls

## 🔗 Useful Links

- [Polymarket](https://polymarket.com/)
- [Polymarket API Docs](https://docs.polymarket.com/)
- [Polars Documentation](https://pola-rs.github.io/polars/)

## 📄 License

MIT License - Free to use and modify

---

**Questions or Issues?**

Check the code comments or run with `--help` flag for more details.
