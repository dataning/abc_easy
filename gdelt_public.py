from google.cloud import bigquery
import pandas as pd

from rich.console import Console
from rich import box, pretty
pretty.install()
console = Console()

client = bigquery.Client.from_service_account_json(
    "gdelt-key.json", project="quickstart-1594651147624"
)

sql = """
SELECT
  GKGRECORDID,
  SourceCommonName,
  DocumentIdentifier AS url,
  -- thematic tags
  Themes,
  V2Themes,
  -- geo tags
  Locations,
  V2Locations,
  -- people
  Persons,
  V2Persons,
  -- orgs
  Organizations,
  V2Organizations,
  -- tone record (a STRUCT of tone metrics)
  V2Tone,
  -- date fields
  Dates,       -- extracted dates mentioned in text
  GCAM,        -- GCAM themes
  -- media embeds
  SharingImage,
  RelatedImages,
  SocialImageEmbeds,
  SocialVideoEmbeds,
  -- quotations & names
  Quotations,
  AllNames,
  -- amounts, translation, extras (XML blob)
  Amounts,
  TranslationInfo,
  Extras,
  -- pull out the <PAGE_TITLE>…</PAGE_TITLE> value
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>([^<]+)</PAGE_TITLE>') 
    AS headline,
  -- partition date as a true DATE
  _PARTITIONDATE AS ingest_date
FROM
  `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
  LOWER(V2Organizations) LIKE "%harvard%"
  AND _PARTITIONDATE = DATE("2025-06-07") AND DATE("2025-06-14")
ORDER BY
  _PARTITIONDATE DESC
"""
df = client.query(sql).to_dataframe()
df

def critical_keyword_filter(df, required_keywords, method='cascading'):
    """
    Cascading approach: Start strict, gradually relax if needed
    """
    results = []
    
    # Level 1: Exact keyword match (highest confidence)
    for keyword in required_keywords:
        exact_mask = df['headline'].str.contains(keyword, case=False, na=False)
        exact_results = df[exact_mask].copy()
        exact_results['match_type'] = f'exact_{keyword}'
        exact_results['confidence'] = 1.0
        results.append(exact_results)
    
    # Level 2: Fuzzy match for typos (medium confidence)
    from fuzzywuzzy import fuzz
    remaining_df = df[~df.index.isin(pd.concat(results).index)]
    
    for keyword in required_keywords:
        fuzzy_scores = remaining_df['headline'].apply(
            lambda x: fuzz.partial_ratio(keyword, str(x)) if pd.notna(x) else 0
        )
        fuzzy_mask = fuzzy_scores >= 85  # Adjust threshold
        fuzzy_results = remaining_df[fuzzy_mask].copy()
        fuzzy_results['match_type'] = f'fuzzy_{keyword}'
        fuzzy_results['confidence'] = fuzzy_scores[fuzzy_mask] / 100
        results.append(fuzzy_results)
    
    return pd.concat(results) if results else pd.DataFrame()

# Usage
critical_results = critical_keyword_filter(df, ["BlackRock"])
critical_results
critical_results.drop_duplicates(subset=['url'], keep='first')

# Comprehensive source analysis
source_stats = df['SourceCommonName'].value_counts()