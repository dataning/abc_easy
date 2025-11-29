"""
iXBRL Parser for SEC 10-K Filings

Extracts structured segment revenue data from inline XBRL tags in SEC HTML filings.
iXBRL (inline XBRL) embeds semantic tags directly in HTML, making it much easier
to extract structured financial data than parsing HTML tables.

Key XBRL concepts for segment data:
- us-gaap:RevenueFromExternalCustomers - Revenue by segment
- us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax - Total revenue
- us-gaap:SegmentReportingInformationRevenue - Segment revenue
- Custom taxonomy concepts (e.g., cat:ConstructionIndustriesSales)
"""

import re
import requests
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class IXBRLParser:
    """Parser for extracting segment data from iXBRL-tagged SEC filings."""
    
    # Revenue-related XBRL concepts (both us-gaap and common custom concepts)
    REVENUE_CONCEPTS = [
        # Standard US-GAAP segment revenue concepts
        'RevenueFromExternalCustomers',
        'RevenuesFromExternalCustomers', 
        'SegmentReportingInformationRevenue',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'Revenues',
        'SalesRevenueNet',
        'SalesRevenueGoodsNet',
        'SalesRevenueServicesNet',
        'NetSales',
        # Segment-specific
        'OperatingSegmentsRevenue',
        'EntityWideRevenueMajorCustomer',
        'RevenueFromContractWithCustomerByGeographicArea',
        'DisaggregationOfRevenueLineItems',
    ]
    
    # Segment dimension concepts
    SEGMENT_DIMENSIONS = [
        'StatementBusinessSegmentsAxis',
        'SegmentReportingSegmentsAxis',
        'StatementOperatingSegmentsAxis',
        'ProductOrServiceAxis',
        'StatementGeographicalAxis',
        'GeographicDistributionForeignAxis',
        'GeographicDistributionDomesticAndForeignAxis',
        'SegmentsAxis',
    ]
    
    def __init__(self, headers: Dict[str, str] = None):
        """Initialize parser with optional custom headers."""
        self.headers = headers or {
            "User-Agent": "IXBRLParser/1.0 (Contact: your-email@example.com)",
            "Accept": "text/html,application/xhtml+xml"
        }
    
    def fetch_filing(self, url: str) -> Optional[str]:
        """Fetch the HTML content of a filing."""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching filing: {e}")
            return None
    
    def parse_filing(self, html_content: str, filing_date: str = None) -> Dict:
        """
        Parse iXBRL tags from an SEC filing HTML document.
        
        Args:
            html_content: The HTML content of the 10-K filing
            filing_date: The filing date for context
            
        Returns:
            Dictionary with extracted segment data
        """
        if not HAS_BS4:
            return {"error": "BeautifulSoup4 required. Install with: pip install beautifulsoup4"}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        result = {
            "filing_date": filing_date,
            "segments": [],
            "geographic": [],
            "products": [],
            "subsegments": [],
            "total_revenue": [],
            "contexts": {},
            "raw_facts": []
        }
        
        # Extract all iXBRL contexts first (they define periods and dimensions)
        result["contexts"] = self._extract_contexts(soup)
        
        # Find all iXBRL tagged elements
        # iXBRL uses tags like: ix:nonFraction, ix:nonNumeric, etc.
        ixbrl_elements = soup.find_all(['ix:nonfraction', 'ix:nonnumeric', 
                                        'ix:fraction', 'ix:continuation'])
        
        # Also check for older XBRL inline format
        if not ixbrl_elements:
            ixbrl_elements = soup.find_all(attrs={'name': re.compile(r'us-gaap:|dei:|.*:')})
        
        # Parse each iXBRL element
        for elem in ixbrl_elements:
            fact = self._parse_ixbrl_element(elem, result["contexts"])
            if fact:
                result["raw_facts"].append(fact)
        
        # Categorize facts into segments, geographic, products
        self._categorize_facts(result)
        
        return result
    
    def _extract_contexts(self, soup: BeautifulSoup) -> Dict:
        """Extract XBRL context definitions which define periods and dimensions."""
        contexts = {}
        
        # Find context elements (in xbrli namespace or regular)
        context_elements = soup.find_all(['xbrli:context', 'context'])
        
        for ctx in context_elements:
            ctx_id = ctx.get('id', '')
            if not ctx_id:
                continue
            
            context_info = {
                "id": ctx_id,
                "period": {},
                "dimensions": {}
            }
            
            # Extract period information
            period = ctx.find(['xbrli:period', 'period'])
            if period:
                instant = period.find(['xbrli:instant', 'instant'])
                start_date = period.find(['xbrli:startdate', 'startdate'])
                end_date = period.find(['xbrli:enddate', 'enddate'])
                
                if instant:
                    context_info["period"]["instant"] = instant.get_text(strip=True)
                if start_date:
                    context_info["period"]["start"] = start_date.get_text(strip=True)
                if end_date:
                    context_info["period"]["end"] = end_date.get_text(strip=True)
            
            # Extract dimension/segment information
            scenario = ctx.find(['xbrli:scenario', 'scenario'])
            segment = ctx.find(['xbrli:segment', 'segment'])
            
            for dim_container in [scenario, segment]:
                if dim_container:
                    # Look for explicit members
                    explicit_members = dim_container.find_all(['xbrldi:explicitmember', 'explicitmember'])
                    for member in explicit_members:
                        dimension = member.get('dimension', '')
                        value = member.get_text(strip=True)
                        if dimension:
                            # Clean up dimension name (remove namespace prefix)
                            dim_name = dimension.split(':')[-1] if ':' in dimension else dimension
                            context_info["dimensions"][dim_name] = value
            
            contexts[ctx_id] = context_info
        
        return contexts
    
    def _parse_ixbrl_element(self, elem, contexts: Dict) -> Optional[Dict]:
        """Parse a single iXBRL tagged element."""
        # Get the concept name
        name = elem.get('name', '')
        if not name:
            return None
        
        # Parse namespace and local name
        if ':' in name:
            namespace, local_name = name.split(':', 1)
        else:
            namespace = 'unknown'
            local_name = name
        
        # Get context reference
        context_ref = elem.get('contextref', '')
        context_info = contexts.get(context_ref, {})
        
        # Get the value
        raw_value = elem.get_text(strip=True)
        
        # Parse numeric value if applicable
        value = self._parse_value(elem, raw_value)
        
        # Get scale (iXBRL uses scale attribute for multiplier)
        scale = elem.get('scale', '0')
        try:
            scale_factor = 10 ** int(scale)
        except ValueError:
            scale_factor = 1
        
        # Get sign (negative indicator)
        sign = elem.get('sign', '')
        if sign == '-' and value is not None:
            value = -value
        
        # Apply scale
        if value is not None:
            value = value * scale_factor
        
        # Get unit reference
        unit_ref = elem.get('unitref', '')
        
        # Get decimals
        decimals = elem.get('decimals', '')
        
        # Extract period from context
        period = context_info.get("period", {})
        fiscal_year = None
        if period.get("end"):
            try:
                end_date = datetime.strptime(period["end"], "%Y-%m-%d")
                fiscal_year = end_date.year
            except ValueError:
                pass
        
        fact = {
            "namespace": namespace,
            "concept": local_name,
            "value": value,
            "raw_value": raw_value,
            "context_ref": context_ref,
            "unit_ref": unit_ref,
            "scale": scale,
            "decimals": decimals,
            "period": period,
            "fiscal_year": fiscal_year,
            "dimensions": context_info.get("dimensions", {})
        }
        
        return fact
    
    def _parse_value(self, elem, raw_value: str) -> Optional[float]:
        """Parse the numeric value from an iXBRL element."""
        # Check format attribute
        format_attr = elem.get('format', '')
        
        # Clean the raw value
        clean_value = raw_value.replace(',', '').replace('$', '').replace(' ', '')
        
        # Handle parentheses as negative
        is_negative = '(' in clean_value and ')' in clean_value
        clean_value = clean_value.replace('(', '').replace(')', '')
        
        # Handle dash as zero or missing
        if clean_value in ['-', '—', '–', '']:
            return None
        
        try:
            value = float(clean_value)
            if is_negative:
                value = -value
            return value
        except ValueError:
            return None
    
    def _categorize_facts(self, result: Dict):
        """Categorize parsed facts into segments, geographic, products, sub-segments, etc."""
        
        segment_data = defaultdict(lambda: defaultdict(dict))  # segment_name -> year -> value
        geographic_data = defaultdict(lambda: defaultdict(dict))
        product_data = defaultdict(lambda: defaultdict(dict))
        subsegment_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # parent_segment -> sub_segment -> year -> value
        total_revenue = defaultdict(dict)  # year -> value
        
        for fact in result["raw_facts"]:
            concept = fact.get("concept", "")
            value = fact.get("value")
            dimensions = fact.get("dimensions", {})
            fiscal_year = fact.get("fiscal_year")
            
            if value is None or fiscal_year is None:
                continue
            
            # Skip negative values (likely adjustments) and tiny values
            if value <= 0:
                continue
            
            # Skip values that are too small to be revenue (likely ratios/percentages)
            # Most segment revenues are in millions at minimum
            if value < 1000000:  # Less than $1M is probably not a segment revenue
                continue
            
            # Check if this is a revenue-related concept
            is_revenue = any(rev_concept.lower() in concept.lower() 
                           for rev_concept in self.REVENUE_CONCEPTS)
            
            # Also check for "Sales" or "Revenue" in concept name
            is_revenue = is_revenue or any(kw in concept.lower() 
                                          for kw in ['sales', 'revenue'])
            
            if not is_revenue:
                continue
            
            # Determine segment name from dimensions
            segment_name = None
            segment_type = None  # 'segment', 'geographic', 'product', 'subsegment'
            parent_segment = None  # For sub-segments, track the parent
            
            # First pass: identify all dimension types present
            has_business_segment = False
            has_major_customer = False
            business_segment_value = None
            major_customer_value = None
            
            for dim_name, dim_value in dimensions.items():
                dim_lower = dim_name.lower()
                value_clean = dim_value.split(':')[-1] if ':' in dim_value else dim_value
                
                if any(seg in dim_lower for seg in ['businesssegment', 'operatingsegment', 
                                                     'reportingsegment', 'segmentsaxis']):
                    has_business_segment = True
                    business_segment_value = value_clean
                elif 'majorcustomer' in dim_lower:
                    has_major_customer = True
                    major_customer_value = value_clean
            
            # If we have both business segment and major customer, it's a sub-segment
            if has_business_segment and has_major_customer:
                parent_segment = self._clean_segment_name(business_segment_value)
                segment_name = self._clean_segment_name(major_customer_value)
                segment_type = 'subsegment'
            else:
                # Original logic for single dimension
                for dim_name, dim_value in dimensions.items():
                    dim_lower = dim_name.lower()
                    value_clean = dim_value.split(':')[-1] if ':' in dim_value else dim_value
                    
                    # Check for business segment dimension
                    if any(seg in dim_lower for seg in ['businesssegment', 'operatingsegment', 
                                                         'reportingsegment', 'segmentsaxis']):
                        segment_name = self._clean_segment_name(value_clean)
                        segment_type = 'segment'
                        break
                    
                    # Check for geographic dimension
                    elif any(geo in dim_lower for geo in ['geographic', 'geographicarea', 
                                                           'geographicdistribution', 'country']):
                        segment_name = self._clean_segment_name(value_clean)
                        segment_type = 'geographic'
                        break
                    
                    # Check for product dimension
                    elif any(prod in dim_lower for prod in ['product', 'service', 
                                                             'productline', 'productorservice']):
                        segment_name = self._clean_segment_name(value_clean)
                        segment_type = 'product'
                        break
                    
                    # Check for major customer as standalone (application breakdown)
                    elif 'majorcustomer' in dim_lower:
                        segment_name = self._clean_segment_name(value_clean)
                        segment_type = 'subsegment'
                        parent_segment = 'Applications'  # Generic parent when no segment specified
                        break
            
            # If no dimension but is revenue concept, might be total revenue
            if segment_name is None and not dimensions:
                if any(total in concept.lower() for total in ['total', 'consolidated', 'revenues']):
                    # Total revenue
                    if fiscal_year not in total_revenue or value > total_revenue[fiscal_year].get('value', 0):
                        total_revenue[fiscal_year] = {
                            'value': value,
                            'concept': concept
                        }
                continue
            
            if segment_name is None:
                continue
            
            # Skip totals, eliminations, and aggregations
            skip_names = ['total', 'consolidated', 'elimination', 'corporate', 
                         'unallocated', 'reconciliation', 'intersegment',
                         'all other', 'aggregation', 'aggregate', 'reportable segment',
                         'adjustments', 'other segment', 'segment total']
            if any(skip in segment_name.lower() for skip in skip_names):
                continue
            
            # Store the value (keep largest if duplicate)
            if segment_type == 'segment':
                if fiscal_year not in segment_data[segment_name] or \
                   value > segment_data[segment_name][fiscal_year].get('value', 0):
                    segment_data[segment_name][fiscal_year] = {
                        'value': value,
                        'concept': concept
                    }
            elif segment_type == 'geographic':
                if fiscal_year not in geographic_data[segment_name] or \
                   value > geographic_data[segment_name][fiscal_year].get('value', 0):
                    geographic_data[segment_name][fiscal_year] = {
                        'value': value,
                        'concept': concept
                    }
            elif segment_type == 'product':
                if fiscal_year not in product_data[segment_name] or \
                   value > product_data[segment_name][fiscal_year].get('value', 0):
                    product_data[segment_name][fiscal_year] = {
                        'value': value,
                        'concept': concept
                    }
            elif segment_type == 'subsegment':
                parent = parent_segment or 'Applications'
                if fiscal_year not in subsegment_data[parent][segment_name] or \
                   value > subsegment_data[parent][segment_name][fiscal_year].get('value', 0):
                    subsegment_data[parent][segment_name][fiscal_year] = {
                        'value': value,
                        'concept': concept
                    }
        
        # Convert to output format
        result["segments"] = self._format_segment_output(segment_data)
        result["geographic"] = self._filter_geo_subtotals(self._format_segment_output(geographic_data))
        result["products"] = self._filter_subtotals(self._format_segment_output(product_data))
        result["subsegments"] = self._format_subsegment_output(subsegment_data)
        result["total_revenue"] = [
            {"fiscal_year": year, "value": data["value"], "concept": data["concept"]}
            for year, data in sorted(total_revenue.items(), reverse=True)
        ]
    
    def _filter_geo_subtotals(self, segments: List[Dict]) -> List[Dict]:
        """
        Filter out geographic subtotals when more detailed breakdowns exist.
        E.g., remove 'Non-US' if 'Europe', 'Asia Pacific', etc. exist.
        """
        # Known geographic subtotal names
        subtotal_patterns = ['non-us', 'nonus', 'international', 'foreign', 
                            'other countries', 'other international', 'rest of world']
        
        # Count non-subtotal segments
        detailed_segments = [s for s in segments 
                           if s["segment_name"].lower() not in subtotal_patterns]
        
        # If we have multiple detailed regions, filter out subtotals
        if len(detailed_segments) >= 3:
            return [s for s in segments 
                   if s["segment_name"].lower() not in subtotal_patterns]
        
        return segments
    
    def _filter_subtotals(self, segments: List[Dict]) -> List[Dict]:
        """
        Filter out subtotal segments when more detailed breakdowns exist.
        E.g., remove 'Product' if 'iPhone', 'iPad', 'Mac' exist.
        """
        # Known subtotal names that should be removed if details exist
        subtotal_patterns = ['product', 'service', 'other', 'service other']
        
        # Count non-subtotal segments
        detailed_segments = [s for s in segments 
                           if s["segment_name"].lower() not in subtotal_patterns]
        
        # If we have detailed segments, filter out the subtotals
        if len(detailed_segments) >= 2:
            return [s for s in segments 
                   if s["segment_name"].lower() not in subtotal_patterns]
        
        return segments
    
    def extract_from_multiple_filings(self, filing_urls: List[Tuple[str, str]]) -> Dict:
        """
        Extract and merge segment data from multiple 10-K filings.
        
        Args:
            filing_urls: List of (url, filing_date) tuples
            
        Returns:
            Merged dictionary with all segment data
        """
        merged_result = {
            "segments": defaultdict(lambda: {}),
            "geographic": defaultdict(lambda: {}),
            "products": defaultdict(lambda: {}),
            "total_revenue": {},
            "filing_dates": []
        }
        
        for url, filing_date in filing_urls:
            result = self.extract_segment_revenue(url, filing_date)
            if "error" in result:
                continue
            
            merged_result["filing_dates"].append(filing_date)
            
            # Merge segments
            for seg in result.get("segments", []):
                name = seg["segment_name"]
                for entry in seg.get("data", []):
                    year = entry["fiscal_year"]
                    if year not in merged_result["segments"][name]:
                        merged_result["segments"][name][year] = entry
            
            # Merge geographic
            for seg in result.get("geographic", []):
                name = seg["segment_name"]
                for entry in seg.get("data", []):
                    year = entry["fiscal_year"]
                    if year not in merged_result["geographic"][name]:
                        merged_result["geographic"][name][year] = entry
            
            # Merge products
            for seg in result.get("products", []):
                name = seg["segment_name"]
                for entry in seg.get("data", []):
                    year = entry["fiscal_year"]
                    if year not in merged_result["products"][name]:
                        merged_result["products"][name][year] = entry
            
            # Merge total revenue
            for entry in result.get("total_revenue", []):
                year = entry["fiscal_year"]
                if year not in merged_result["total_revenue"]:
                    merged_result["total_revenue"][year] = entry
        
        # Convert back to list format
        final_result = {
            "filing_dates": merged_result["filing_dates"],
            "segments": [],
            "geographic": [],
            "products": [],
            "total_revenue": [],
            "contexts": {},
            "raw_facts": []
        }
        
        for name, years in merged_result["segments"].items():
            final_result["segments"].append({
                "segment_name": name,
                "data": sorted(years.values(), key=lambda x: x["fiscal_year"], reverse=True)
            })
        
        for name, years in merged_result["geographic"].items():
            final_result["geographic"].append({
                "segment_name": name,
                "data": sorted(years.values(), key=lambda x: x["fiscal_year"], reverse=True)
            })
        
        for name, years in merged_result["products"].items():
            final_result["products"].append({
                "segment_name": name,
                "data": sorted(years.values(), key=lambda x: x["fiscal_year"], reverse=True)
            })
        
        final_result["total_revenue"] = sorted(
            merged_result["total_revenue"].values(),
            key=lambda x: x["fiscal_year"],
            reverse=True
        )
        
        # Sort segment lists
        final_result["segments"].sort(key=lambda x: x["segment_name"])
        final_result["geographic"].sort(key=lambda x: x["segment_name"])
        final_result["products"].sort(key=lambda x: x["segment_name"])
        
        return final_result
    
    def _clean_segment_name(self, name: str) -> str:
        """Clean up segment name from XBRL member values."""
        # Remove common suffixes
        name = re.sub(r'Member$', '', name)
        name = re.sub(r'Segment$', '', name)
        name = re.sub(r'Domain$', '', name)
        name = re.sub(r'Operations$', '', name)
        
        # Convert CamelCase to spaces (but be careful with abbreviations)
        # Insert space before uppercase letters that follow lowercase
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        # Insert space before uppercase letters in sequences like "EMEAMember"
        name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
        
        # Fix common issues - company-specific patterns
        # Order matters - do longer patterns first
        replacements = [
            ('Three Six Five', '365'),
            ('Linked In', 'LinkedIn'),
            ('I Phone', 'iPhone'),
            ('I Pad', 'iPad'),
            ('Energyand', 'Energy and'),
            ('Homeand', 'Home and'),
            ('E M E A', 'EMEA'),
            ('Non Us', 'Non-US'),
            ('U S', 'US'),
            ('Oil And Gas Customer', 'Oil and Gas'),
            ('Powergeneration', 'Power Generation'),
        ]
        
        for old, new in replacements:
            name = name.replace(old, new)
        
        # Clean up extra spaces
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        return name
    
    def _format_segment_output(self, segment_data: Dict) -> List[Dict]:
        """Format segment data for output."""
        output = []
        
        for segment_name, years in segment_data.items():
            segment_entry = {
                "segment_name": segment_name,
                "data": []
            }
            
            for year, info in sorted(years.items(), reverse=True):
                segment_entry["data"].append({
                    "fiscal_year": year,
                    "value": info["value"],
                    "concept": info["concept"]
                })
            
            output.append(segment_entry)
        
        # Sort segments by name
        output.sort(key=lambda x: x["segment_name"])
        
        return output
    
    def _format_subsegment_output(self, subsegment_data: Dict) -> List[Dict]:
        """Format sub-segment data for output (e.g., E&T applications)."""
        output = []
        
        for parent_segment, sub_segments in subsegment_data.items():
            parent_entry = {
                "parent_segment": parent_segment,
                "subsegments": []
            }
            
            for sub_name, years in sub_segments.items():
                sub_entry = {
                    "segment_name": sub_name,
                    "data": []
                }
                
                for year, info in sorted(years.items(), reverse=True):
                    sub_entry["data"].append({
                        "fiscal_year": year,
                        "value": info["value"],
                        "concept": info["concept"]
                    })
                
                parent_entry["subsegments"].append(sub_entry)
            
            # Sort sub-segments by name
            parent_entry["subsegments"].sort(key=lambda x: x["segment_name"])
            output.append(parent_entry)
        
        # Sort by parent segment name
        output.sort(key=lambda x: x["parent_segment"])
        
        return output
    
    def extract_segment_revenue(self, url: str, filing_date: str = None) -> Dict:
        """
        Convenience method to fetch and parse a filing in one call.
        
        Args:
            url: URL of the SEC filing HTML document
            filing_date: Optional filing date
            
        Returns:
            Dictionary with extracted segment revenue data
        """
        html_content = self.fetch_filing(url)
        if not html_content:
            return {"error": "Could not fetch filing"}
        
        return self.parse_filing(html_content, filing_date)


def format_currency(value: float) -> str:
    """Format a number as currency."""
    if abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"


def print_ixbrl_report(data: Dict, verbose: bool = False):
    """Print a formatted report of iXBRL extracted data."""
    print("\n" + "=" * 85)
    print("iXBRL SEGMENT REVENUE EXTRACTION")
    print("=" * 85)
    
    if data.get("filing_date"):
        print(f"Filing Date: {data['filing_date']}")
    
    name_width = 35
    val_width = 12
    
    # Print business segments
    if data.get("segments"):
        print("\n📊 BUSINESS SEGMENTS:")
        print("-" * 70)
        
        # Collect all years
        all_years = set()
        for seg in data["segments"]:
            for d in seg.get("data", []):
                all_years.add(d.get("fiscal_year"))
        
        sorted_years = sorted(all_years, reverse=True)[:5]
        
        # Header
        header = f"{'Segment':<{name_width}}"
        for year in sorted_years:
            header += f"{'FY' + str(year):>{val_width}}"
        print(header)
        print("-" * (name_width + val_width * len(sorted_years)))
        
        # Data rows
        for seg in data["segments"]:
            name = seg["segment_name"][:name_width-2]
            row = f"{name:<{name_width}}"
            year_values = {d["fiscal_year"]: d["value"] for d in seg.get("data", [])}
            
            for year in sorted_years:
                if year in year_values:
                    row += f"{format_currency(year_values[year]):>{val_width}}"
                else:
                    row += f"{'-':>{val_width}}"
            print(row)
    
    # Print geographic segments
    if data.get("geographic"):
        print("\n🌍 GEOGRAPHIC SEGMENTS:")
        print("-" * 70)
        
        all_years = set()
        for seg in data["geographic"]:
            for d in seg.get("data", []):
                all_years.add(d.get("fiscal_year"))
        
        sorted_years = sorted(all_years, reverse=True)[:5]
        
        header = f"{'Region':<{name_width}}"
        for year in sorted_years:
            header += f"{'FY' + str(year):>{val_width}}"
        print(header)
        print("-" * (name_width + val_width * len(sorted_years)))
        
        for seg in data["geographic"]:
            name = seg["segment_name"][:name_width-2]
            row = f"{name:<{name_width}}"
            year_values = {d["fiscal_year"]: d["value"] for d in seg.get("data", [])}
            
            for year in sorted_years:
                if year in year_values:
                    row += f"{format_currency(year_values[year]):>{val_width}}"
                else:
                    row += f"{'-':>{val_width}}"
            print(row)
    
    # Print product segments
    if data.get("products"):
        print("\n📦 PRODUCT/SERVICE SEGMENTS:")
        print("-" * 70)
        
        all_years = set()
        for seg in data["products"]:
            for d in seg.get("data", []):
                all_years.add(d.get("fiscal_year"))
        
        sorted_years = sorted(all_years, reverse=True)[:5]
        
        header = f"{'Product/Service':<{name_width}}"
        for year in sorted_years:
            header += f"{'FY' + str(year):>{val_width}}"
        print(header)
        print("-" * (name_width + val_width * len(sorted_years)))
        
        for seg in data["products"]:
            name = seg["segment_name"][:name_width-2]
            row = f"{name:<{name_width}}"
            year_values = {d["fiscal_year"]: d["value"] for d in seg.get("data", [])}
            
            for year in sorted_years:
                if year in year_values:
                    row += f"{format_currency(year_values[year]):>{val_width}}"
                else:
                    row += f"{'-':>{val_width}}"
            print(row)
    
    # Print sub-segments (e.g., E&T applications)
    if data.get("subsegments"):
        print("\n🔧 SUB-SEGMENT BREAKDOWN:")
        print("-" * 70)
        
        for parent_group in data["subsegments"]:
            parent_name = parent_group.get("parent_segment", "Unknown")
            subsegments = parent_group.get("subsegments", [])
            
            if not subsegments:
                continue
            
            # Get all years from subsegments
            all_years = set()
            for seg in subsegments:
                for d in seg.get("data", []):
                    all_years.add(d.get("fiscal_year"))
            
            sorted_years = sorted(all_years, reverse=True)[:5]
            
            print(f"\n  {parent_name}:")
            header = f"  {'Application':<{name_width-2}}"
            for year in sorted_years:
                header += f"{'FY' + str(year):>{val_width}}"
            print(header)
            print("  " + "-" * (name_width - 2 + val_width * len(sorted_years)))
            
            for seg in subsegments:
                name = seg["segment_name"][:name_width-4]
                row = f"  {name:<{name_width-2}}"
                year_values = {d["fiscal_year"]: d["value"] for d in seg.get("data", [])}
                
                for year in sorted_years:
                    if year in year_values:
                        row += f"{format_currency(year_values[year]):>{val_width}}"
                    else:
                        row += f"{'-':>{val_width}}"
                print(row)
    
    # Print total revenue
    if data.get("total_revenue"):
        print("\n💰 TOTAL REVENUE:")
        for entry in data["total_revenue"][:5]:
            print(f"   FY{entry['fiscal_year']}: {format_currency(entry['value'])}")
    
    # Summary stats
    total_facts = len(data.get("raw_facts", []))
    total_contexts = len(data.get("contexts", {}))
    
    # Count total sub-segments
    subsegment_count = sum(len(p.get("subsegments", [])) for p in data.get("subsegments", []))
    
    print(f"\n📈 Extraction Stats:")
    print(f"   Total iXBRL facts parsed: {total_facts}")
    print(f"   Total contexts found: {total_contexts}")
    print(f"   Business segments: {len(data.get('segments', []))}")
    print(f"   Geographic segments: {len(data.get('geographic', []))}")
    print(f"   Product segments: {len(data.get('products', []))}")
    if subsegment_count > 0:
        print(f"   Sub-segments (applications): {subsegment_count}")
    
    if verbose and data.get("raw_facts"):
        print("\n📝 SAMPLE RAW FACTS (first 20 revenue-related):")
        count = 0
        for fact in data["raw_facts"]:
            if count >= 20:
                break
            if fact.get("value") and any(kw in fact.get("concept", "").lower() 
                                         for kw in ['revenue', 'sales']):
                dims = fact.get("dimensions", {})
                dim_str = ", ".join(f"{k}={v}" for k, v in dims.items()) if dims else "no dimensions"
                print(f"   {fact['concept']}: {format_currency(fact['value'])} "
                      f"(FY{fact.get('fiscal_year', '?')}) [{dim_str}]")
                count += 1
    
    print("\n" + "=" * 85)


def main():
    """Demo the iXBRL parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse iXBRL segment data from SEC filings")
    parser.add_argument("url", help="URL of the SEC 10-K HTML filing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show raw facts")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if not HAS_BS4:
        print("Error: BeautifulSoup4 required. Install with: pip install beautifulsoup4")
        return
    
    print(f"Fetching and parsing: {args.url}")
    
    ixbrl_parser = IXBRLParser()
    result = ixbrl_parser.extract_segment_revenue(args.url)
    
    if args.json:
        import json
        # Remove raw_facts for cleaner JSON output unless verbose
        if not args.verbose:
            result.pop("raw_facts", None)
            result.pop("contexts", None)
        print(json.dumps(result, indent=2, default=str))
    else:
        print_ixbrl_report(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
