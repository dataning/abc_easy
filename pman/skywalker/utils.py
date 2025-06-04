def parse_gdelt_date(date_str):
    """
    Convert GDELT date format to datetime
    
    Args:
        date_str (str): Date string in GDELT format (YYYYMMDDThhmmssZ)
        
    Returns:
        str: Formatted date string (YYYY-MM-DD)
    """
    import datetime
    try:
        # Remove 'Z' and 'T' characters
        date_str = date_str.replace('Z', '').replace('T', '')
        # Parse date
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(date_str[8:10]) if len(date_str) > 8 else 0
        minute = int(date_str[10:12]) if len(date_str) > 10 else 0
        second = int(date_str[12:14]) if len(date_str) > 12 else 0
        
        dt = datetime.datetime(year, month, day, hour, minute, second)
        return dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error parsing date: {e}")
        return date_str

def count_articles_by_source(df):
    """
    Count number of articles by source domain
    
    Args:
        df (pandas.DataFrame): DataFrame containing GDELT articles
        
    Returns:
        pandas.DataFrame: Sorted DataFrame with counts by domain
    """
    if 'domain' not in df.columns:
        raise ValueError("DataFrame must contain 'domain' column")
    
    domain_counts = df['domain'].value_counts().reset_index()
    domain_counts.columns = ['Domain', 'Count']
    return domain_counts