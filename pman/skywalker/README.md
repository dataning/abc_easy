"""
# Skywalker

A Python package for retrieving and saving news articles from the GDELT API.

## Installation

```bash
pip install skywalker
```

Or install from source:

```bash
git clone https://github.com/yourusername/skywalker.git
cd skywalker
pip install -e .
```

## Usage

### Basic Usage

```python
from skywalker import get_and_save_gdelt_articles

# Get articles about Harvard's endowment
articles_df = get_and_save_gdelt_articles(
    query="harvard endowment", 
    max_records=250
)
```

### With Filtering

```python
# Get articles from specific news sources
financial_news_df = get_and_save_gdelt_articles(
    query="harvard endowment", 
    domain=["wsj.com", "bloomberg.com", "ft.com"],
    timespan="90d",  # last 90 days
    output_file="harvard_endowment_financial_news.parquet"
)

# Get articles from a specific country and language
us_news_df = get_and_save_gdelt_articles(
    query="harvard endowment", 
    start_date="2023-01-01", 
    end_date="2023-12-31",
    country="US",
    language="en",
    output_file="harvard_endowment_us_2023.parquet"
)

# Search for exact phrase
exact_df = get_and_save_gdelt_articles(
    query="harvard endowment",
    exact_phrase=True,
    max_records=200
)
```

## Parameters

- `query` (str or list): Search query (e.g., "harvard endowment" or ["harvard", "endowment"])
- `output_file` (str): Output parquet file path. If None, uses query name
- `max_records` (int): Maximum number of records to return
- `sort` (str): Sorting method - "datedesc" (newest first) or "hybridrel" (relevance)
- `start_date` (str): Start date in YYYY-MM-DD format
- `end_date` (str): End date in YYYY-MM-DD format
- `timespan` (str): Alternative to start/end dates (e.g., "30d" for last 30 days)
- `domain` (str or list): Filter by domain(s) (e.g., "nytimes.com" or ["nytimes.com", "wsj.com"])
- `country` (str or list): Filter by source country using FIPS 2-letter code(s) (e.g., "US" or ["US", "GB"])
- `language` (str or list): Filter by language using ISO 639 code(s) (e.g., "en" or ["en", "es"])
- `exact_phrase` (bool): Whether to search for the exact phrase (if query is a string)

## License

MIT
"""