import pandas as pd
import requests
import urllib.parse
import json

def get_and_save_gdelt_articles(query, output_file=None, max_records=250, sort="datedesc", 
                         start_date=None, end_date=None, timespan=None, 
                         domain=None, country=None, language=None, exact_phrase=False):
    """
    Get article data from GDELT API using JSON format, process it, and save to a Parquet file
    
    Parameters:
    query (str or list): Search query (e.g., "harvard endowment" or ["harvard", "endowment"])
    output_file (str): Output parquet file path. If None, uses query name
    max_records (int): Maximum number of records to return
    sort (str): Sorting method - "datedesc" (newest first) or "hybridrel" (relevance)
    start_date (str): Start date in YYYY-MM-DD format
    end_date (str): End date in YYYY-MM-DD format
    timespan (str): Alternative to start/end dates (e.g., "30d" for last 30 days)
    domain (str or list): Filter by domain(s) (e.g., "nytimes.com" or ["nytimes.com", "wsj.com"])
    country (str or list): Filter by source country using FIPS 2-letter code(s) (e.g., "US" or ["US", "GB"])
    language (str or list): Filter by language using ISO 639 code(s) (e.g., "en" or ["en", "es"])
    exact_phrase (bool): Whether to search for the exact phrase (if query is a string)
    
    Returns:
    pandas.DataFrame: Processed DataFrame containing article information
    """
    # Build query parameters
    query_params = []
    
    # Add main keyword query
    if isinstance(query, str):
        if exact_phrase:
            # Search for exact phrase
            query_params.append(f'"{query}"')
        else:
            # Search for terms without requiring exact phrase
            query_params.append(f'{query}')
    elif isinstance(query, list):
        query_terms = [f'"{term}"' if ' ' in term else term for term in query]
        query_params.append(f"({' OR '.join(query_terms)})")
    
    # Add domain filter
    if domain:
        if isinstance(domain, str):
            query_params.append(f"domain:{domain}")
        elif isinstance(domain, list):
            domain_terms = [f"domain:{d}" for d in domain]
            query_params.append(f"({' OR '.join(domain_terms)})")
    
    # Add country filter
    if country:
        if isinstance(country, str):
            query_params.append(f"sourcecountry:{country}")
        elif isinstance(country, list):
            country_terms = [f"sourcecountry:{c}" for c in country]
            query_params.append(f"({' OR '.join(country_terms)})")
    
    # Add language filter
    if language:
        if isinstance(language, str):
            query_params.append(f"sourcelang:{language}")
        elif isinstance(language, list):
            language_terms = [f"sourcelang:{l}" for l in language]
            query_params.append(f"({' OR '.join(language_terms)})")
    
    # Join all query parameters
    full_query = " ".join(query_params)
    encoded_query = urllib.parse.quote_plus(full_query)
    
    # Base URL with format and mode
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?format=json&mode=artlist&maxrecords={max_records}&sort={sort}"
    
    # Add date parameters
    if start_date and end_date:
        url += f"&startdatetime={start_date}&enddatetime={end_date}"
    elif timespan:
        url += f"&timespan={timespan}"
    else:
        url += "&timespan=FULL"  # Default to all time if no date parameters provided
    
    # Add query
    url += f"&query={encoded_query}"
    
    # Set headers to mimic browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) Gecko/20100101 Firefox/137.0",
        "Accept": "*/*"
    }
    
    print(f"Retrieving articles for: {query} (sorted by {'date' if sort=='datedesc' else 'relevance'})")
    
    # Get data as JSON
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Parse JSON response
            data = json.loads(response.text)
            
            # Check if articles exist in the response
            if "articles" in data:
                df = pd.DataFrame(data["articles"])
                print(f"Retrieved {len(df)} articles")
            else:
                print("No articles found in response")
                return pd.DataFrame()
        else:
            raise Exception(f"Failed to retrieve articles. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error retrieving data: {str(e)}")
        return pd.DataFrame()
    
    # Process the data
    print("Processing article data...")
    
    # Convert seendate to datetime (column name in JSON response)
    if 'seendate' in df.columns:
        df['Date'] = pd.to_datetime(df['seendate'])
    
    # Use domain from response if available, otherwise extract from URL
    if 'domain' not in df.columns:
        df['domain'] = df['url'].str.extract(r'https?://(?:www\.)?([^/]+)')
    
    # Create year and month columns for easier analysis
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # First convert to UTC, then remove timezone info before converting to period
    df['YearMonth'] = df['Date'].dt.tz_convert('UTC').dt.tz_localize(None).dt.to_period('M')
    
    # Double-check sorting (in case API doesn't sort correctly)
    if sort == "datedesc":
        df = df.sort_values('Date', ascending=False).reset_index(drop=True)
    
    # Print column names to show what's available
    print(f"Available columns: {df.columns.tolist()}")
    
    # Save to parquet if output_file is specified
    if output_file is not None:
        print(f"Saving to {output_file}...")
        df.to_parquet(output_file, index=False)
        print(f"Saved {len(df)} articles to Parquet file: {output_file}")
    elif isinstance(query, str):
        # Generate filename from query if it's a string
        default_file = f"{query.replace(' ', '_').lower()}_articles.parquet"
        print(f"Saving to {default_file}...")
        df.to_parquet(default_file, index=False)
        print(f"Saved {len(df)} articles to Parquet file: {default_file}")
    
    # Display sample with dates
    print("\nFirst 5 articles (sorted by date):")
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        date_str = row['Date'].strftime('%Y-%m-%d')
        print(f"{i+1}. [{date_str}] {row['title']} ({row['domain']})")
    
    return df