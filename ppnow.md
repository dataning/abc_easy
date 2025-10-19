import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from alog_nav import (
    EstimationData, EstimationLogscaleParameters, EstimationInputs,
    RollForward, NAVcasting, Estimate
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="NAVcasting Production App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black-themed design
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #FFFFFF;
    }
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #000000;
        font-weight: 600;
    }
    .stMarkdown {
        color: #000000;
    }
    
    /* Remove default red theme - Buttons */
    .stButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #333333 !important;
        border-color: #333333 !important;
    }
    .stButton > button:active {
        background-color: #000000 !important;
        border-color: #000000 !important;
    }
    
    /* Slider - clean black theme with visible track */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Slider container */
    [data-baseweb="slider"] {
        height: 6px;
    }
    
    /* Slider track (gray background) */
    [data-baseweb="slider"] > div:nth-child(1) {
        background: #D0D0D0 !important;
        height: 6px !important;
        border-radius: 3px;
    }
    
    /* Slider filled track (black progress) */
    [data-baseweb="slider"] > div:nth-child(2) {
        background: #000000 !important;
        height: 6px !important;
        border-radius: 3px;
    }
    
    /* Slider thumb (visible black circle) */
    [data-baseweb="slider"] [role="slider"] {
        background-color: #000000 !important;
        width: 18px !important;
        height: 18px !important;
        border-radius: 50% !important;
        border: 2px solid #000000 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        cursor: pointer !important;
    }
    
    [data-baseweb="slider"] [role="slider"]:hover {
        background-color: #000000 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
        transform: scale(1.1);
    }
    
    [data-baseweb="slider"] [role="slider"]:focus {
        background-color: #000000 !important;
        box-shadow: 0 0 0 3px rgba(0,0,0,0.1) !important;
        outline: none !important;
    }
    
    /* Alternative selectors for older Streamlit versions */
    .stSlider > div > div > div {
        height: 6px;
    }
    
    .stSlider > div > div > div > div:nth-child(1) {
        background: #D0D0D0 !important;
        height: 6px !important;
    }
    
    .stSlider > div > div > div > div:nth-child(2) {
        background: #000000 !important;
        height: 6px !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: #000000 !important;
        width: 18px !important;
        height: 18px !important;
        border: 2px solid #000000 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    /* Radio buttons */
    .stRadio > label > div[role="radiogroup"] > label > div:first-child {
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
    }
    .stRadio > label > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #000000 !important;
    }
    
    /* Checkboxes */
    .stCheckbox > label > div[role="checkbox"] {
        border: 2px solid #000000 !important;
    }
    .stCheckbox > label > div[role="checkbox"][data-checked="true"] {
        background-color: #000000 !important;
    }
    
    /* Select boxes - remove red focus */
    .stSelectbox > div > div {
        border-color: #CCCCCC !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        border-color: #CCCCCC !important;
    }
    .stSelectbox [data-baseweb="select"]:focus-within > div {
        border-color: #000000 !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 0, 0, 0.15) !important;
    }
    .stSelectbox [data-baseweb="select"] > div:hover {
        border-color: #000000 !important;
    }
    
    /* Multiselect - remove red focus */
    .stMultiSelect > div > div {
        border-color: #CCCCCC !important;
    }
    .stMultiSelect > div > div:focus-within {
        border-color: #000000 !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 0, 0, 0.15) !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: #FFFFFF !important;
    }
    
    /* Text inputs - remove red borders */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        border-color: #CCCCCC !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stDateInput > div > div > input:focus {
        border-color: #000000 !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 0, 0, 0.15) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input:focus-visible,
    .stNumberInput > div > div > input:focus-visible,
    .stDateInput > div > div > input:focus-visible {
        border-color: #000000 !important;
        outline: none !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 2px solid #000000 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #333333 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #F5F5F5 !important;
        border: 1px solid #000000 !important;
        color: #000000 !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: #E8E8E8 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F5F5F5;
        border: 1px solid #000000;
        color: #000000;
    }
    .stTabs [aria-selected="true"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #000000 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #000000 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #000000;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #333333;
        color: #FFFFFF;
    }
    
    /* Production badge */
    .production-badge {
        background: linear-gradient(135deg, #000000 0%, #333333 100%);
        color: #FFFFFF;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SECTOR & COMPANY DATA (same as original)
# ============================================================================

SECTORS = {
    "Technology": {
        "companies": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ADBE", "CRM", "ORCL"],
        "indices": ["QQQ", "XLK", "IGV", "SKYY"]
    },
    "Healthcare": {
        "companies": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "DHR", "BMY"],
        "indices": ["XLV", "IBB", "IHI", "IHF"]
    },
    "Financial Services": {
        "companies": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW"],
        "indices": ["XLF", "KBE", "KRE", "IAT"]
    },
    "Consumer Discretionary": {
        "companies": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "LOW", "TGT"],
        "indices": ["XLY", "VCR", "RTH", "FXD"]
    },
    "Energy": {
        "companies": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO"],
        "indices": ["XLE", "VDE", "IYE", "IXC"]
    },
    "Real Estate": {
        "companies": ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "WELL", "DLR"],
        "indices": ["XLRE", "VNQ", "IYR", "RWR"]
    },
    "Industrials": {
        "companies": ["HON", "UNP", "UPS", "CAT", "BA", "GE", "MMM", "LMT"],
        "indices": ["XLI", "VIS", "IYJ", "FIDU"]
    },
    "Materials": {
        "companies": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "DD"],
        "indices": ["XLB", "VAW", "IYM", "RTM"]
    }
}

# ============================================================================
# HELPER FUNCTIONS (same as original)
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, start_date, end_date, data_type="price"):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if len(tickers) == 1:
            if 'Close' not in data.columns:
                st.error(f"Could not fetch data for {tickers[0]}")
                return None
            result = pd.DataFrame({tickers[0]: data['Close']})
        else:
            if 'Close' not in data.columns.get_level_values(0):
                st.error("Could not fetch data")
                return None
            result = data['Close']
        
        result = result.dropna(how='all')
        
        if result.empty:
            st.error("No data available for the selected period")
            return None
            
        return result
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def generate_synthetic_gp_marks(dates, base_value=100, volatility=0.15, num_marks=2):
    """Generate synthetic GP marks with realistic patterns"""
    np.random.seed(42)
    
    gp_marks = pd.DataFrame(index=dates)
    
    for i in range(num_marks):
        trend = np.linspace(0, 0.3, len(dates))
        noise = np.random.normal(0, volatility, len(dates))
        values = base_value * np.exp(trend + noise.cumsum() * 0.1)
        
        # Create alternating pattern mask based on actual length
        if i == 0:
            mask = np.array([j % 2 == 0 for j in range(len(dates))])  # True, False, True, False...
        else:
            mask = np.array([j % 2 == 1 for j in range(len(dates))])  # False, True, False, True...
        
        values[~mask] = np.nan
        gp_marks[f"GP_Mark_{i+1}"] = values
    
    if gp_marks.iloc[0].isna().all():
        gp_marks.iloc[0, 0] = base_value
    
    return gp_marks


def resample_to_quarterly(data, method='last'):
    """Resample data to quarterly frequency"""
    if method == 'last':
        return data.resample('QE').last()
    elif method == 'mean':
        return data.resample('QE').mean()
    else:
        return data.resample('QE').last()


def calculate_correlations(private_proxy, comparables):
    """Calculate correlations between private company proxy and comparables"""
    log_returns_private = np.log(private_proxy).diff().dropna()
    log_returns_comps = np.log(comparables).diff().dropna()
    
    common_dates = log_returns_private.index.intersection(log_returns_comps.index)
    log_returns_private = log_returns_private.loc[common_dates]
    log_returns_comps = log_returns_comps.loc[common_dates]
    
    correlations = {}
    raw_correlations = {}
    for col in log_returns_comps.columns:
        raw_corr = log_returns_private.corr(log_returns_comps[col])
        raw_correlations[col] = raw_corr
        correlations[col] = max(0.3, min(0.95, raw_corr))
    
    return pd.Series(correlations), pd.Series(raw_correlations)


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Production badge
    st.markdown('<div class="production-badge">🎯 Full Version</div>', unsafe_allow_html=True)
    
    st.title("NAVcasting Private Company Valuation")
    st.markdown("""
    <div style='color: #666666; font-size: 16px; margin-bottom: 30px;'>
    <strong>Production-Grade Implementation</strong>
    </div>
    """, unsafe_allow_html=True)

    # ========================================================================
    # STEP 1: DATE RANGE SELECTION (matching standard app layout)
    # ========================================================================
    
    st.markdown("<h2 style='color: #000000;'>1. Date Range</h2>", unsafe_allow_html=True)
    
    # Generate list of quarter end dates
    def generate_quarter_ends(start_year, end_year):
        """Generate list of all quarter end dates"""
        quarter_ends = []
        for year in range(start_year, end_year + 1):
            quarter_ends.append(datetime(year, 3, 31).date())
            quarter_ends.append(datetime(year, 6, 30).date())
            quarter_ends.append(datetime(year, 9, 30).date())
            quarter_ends.append(datetime(year, 12, 31).date())
        return quarter_ends
    
    # Get quarter ends for past 10 years to today
    current_year = datetime.now().year
    all_quarter_ends = generate_quarter_ends(current_year - 10, current_year)
    all_quarter_ends = [qe for qe in all_quarter_ends if qe <= datetime.now().date()]
    
    # Default values
    default_start = datetime(current_year - 3, 12, 31).date()
    default_end = all_quarter_ends[-1]  # Most recent quarter end
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_date = st.selectbox(
            "Start Date (Quarter End)",
            options=all_quarter_ends,
            index=all_quarter_ends.index(default_start) if default_start in all_quarter_ends else len(all_quarter_ends) - 13,
            format_func=lambda x: x.strftime('%Y-%m-%d')
        )
    with col2:
        # Filter end dates to only show those after start date
        valid_end_dates = [qe for qe in all_quarter_ends if qe >= start_date]
        end_date = st.selectbox(
            "End Date (Quarter End)",
            options=valid_end_dates,
            index=len(valid_end_dates) - 1,
            format_func=lambda x: x.strftime('%Y-%m-%d')
        )
    with col3:
        years = (end_date - start_date).days // 365
        st.markdown(f"""
        <div style='background-color: #000000; padding: 15px; border-radius: 8px; border-left: 3px solid #000000;'>
            <div style='color: #CCCCCC; font-size: 12px;'>TIME SPAN</div>
            <div style='color: #FFFFFF; font-size: 24px; font-weight: 600;'>{years}+ years</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Generate quarterly dates for the rest of the app
    quarterly_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    
    st.markdown("<hr style='border: 1px solid #000000; margin: 30px 0;'>", unsafe_allow_html=True)
    
    # ========================================================================
    # STEP 2: PRIVATE COMPANY SETUP (matching standard app layout)
    # ========================================================================
    
    st.markdown("<h2 style='color: #000000;'>2. Private Company Setup</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("Company Name", "TechCo Private Ltd.")
    with col2:
        initial_value = st.number_input(
            "Initial Valuation ($M)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )
    
    st.markdown("<h3 style='color: #000000; margin-top: 20px;'>GP Marks Input Method</h3>", unsafe_allow_html=True)
    
    gp_input_method = st.selectbox(
        "How would you like to provide GP marks?",
        ["Generate Synthetic (Demo)", "Upload CSV File", "Manual Entry"],
        index=0
    )
    
    gp_marks = None
    num_gp_marks = 2
    
    if gp_input_method == "Upload CSV File":
        st.markdown("""
        <div style='background-color: #F5F5F5; padding: 15px; border-radius: 8px; color: #000000;'>
        <strong>CSV Format Requirements:</strong>
        <ul>
        <li>First column: <code>date</code> (YYYY-MM-DD format, quarter-ends recommended)</li>
        <li>Subsequent columns: Each GP mark source (e.g., 'Fund_Manager_A', 'Independent_Appraisal')</li>
        <li>Use blank cells or NaN for missing valuations</li>
        <li>First date must have at least one non-missing value</li>
        </ul>
        Example: <code>date,GP_Mark_1,GP_Mark_2</code>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_gp = st.file_uploader(
            "Upload GP Marks CSV",
            type=['csv'],
            help="CSV with date column and GP mark sources as columns"
        )
        
        if uploaded_gp:
            try:
                gp_marks = pd.read_csv(uploaded_gp, parse_dates=['date'], index_col='date')
                gp_marks.index = pd.to_datetime(gp_marks.index)
                gp_marks = gp_marks.reindex(quarterly_dates)
                num_gp_marks = len(gp_marks.columns)
                st.success(f"✅ Loaded {num_gp_marks} GP mark sources")
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                st.stop()
        else:
            st.info("📤 Please upload a CSV file to continue")
            st.stop()
    
    elif gp_input_method == "Manual Entry":
        num_gp_marks = st.number_input("Number of GP Mark Sources", min_value=1, max_value=4, value=2, step=1)
        
        st.markdown("<p style='color: #666666;'>Enter GP marks for key dates (leave blank if unavailable):</p>", unsafe_allow_html=True)
        
        # Generate quarterly dates
        quarters = pd.date_range(start_date, end_date, freq='QE')
        manual_gp_data = {f"GP_Mark_{i+1}": [] for i in range(num_gp_marks)}
        
        # Only show every other quarter for brevity, or all if < 8 quarters
        display_quarters = quarters if len(quarters) <= 8 else quarters[::2]
        
        for quarter in display_quarters:
            cols = st.columns([2] + [1] * num_gp_marks)
            with cols[0]:
                st.markdown(f"<p style='color: #000000; padding-top: 10px;'>{quarter.strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)
            for i in range(num_gp_marks):
                with cols[i + 1]:
                    val = st.number_input(
                        f"Mark {i+1}",
                        min_value=0.0,
                        value=0.0,
                        step=10.0,
                        key=f"manual_{quarter}_{i}",
                        label_visibility="collapsed"
                    )
                    manual_gp_data[f"GP_Mark_{i+1}"].append(val if val > 0 else np.nan)
        
        # Create full dataframe (fill missing dates with NaN)
        full_gp_data = {f"GP_Mark_{i+1}": [] for i in range(num_gp_marks)}
        manual_idx = 0
        for quarter in quarters:
            if quarter in display_quarters:
                for i in range(num_gp_marks):
                    full_gp_data[f"GP_Mark_{i+1}"].append(manual_gp_data[f"GP_Mark_{i+1}"][manual_idx])
                manual_idx += 1
            else:
                for i in range(num_gp_marks):
                    full_gp_data[f"GP_Mark_{i+1}"].append(np.nan)
        
        gp_marks = pd.DataFrame(full_gp_data, index=quarters)
        
        if gp_marks.iloc[0].notna().any():
            st.success("✅ GP marks entered successfully")
        else:
            st.warning("⚠️ First period must have at least one GP mark")
            st.stop()
    
    else:  # Generate Synthetic
        st.warning("⚠️ Using synthetic data for demonstration only. Upload real GP marks for production use.")
        num_gp_marks = st.number_input("Number of GP Mark Sources", min_value=1, max_value=4, value=2, step=1)
        gp_marks = generate_synthetic_gp_marks(quarterly_dates, base_value=initial_value, num_marks=num_gp_marks)
        st.success("✅ Generated synthetic GP marks")
    
    # Optional: 3rd Party Valuation Benchmark
    st.markdown("<h3 style='color: #000000; margin-top: 20px;'>3rd Party Valuation Benchmark (Optional)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666666;'>Compare model results against independent appraisals or external valuations</p>", unsafe_allow_html=True)
    
    use_benchmark = st.checkbox("Compare against 3rd party valuation", value=False)
    benchmark_data = None
    
    if use_benchmark:
        col1, col2, col3 = st.columns(3)
        with col1:
            benchmark_date = st.selectbox(
                "Benchmark Date",
                options=quarterly_dates,
                index=len(quarterly_dates) - 1,
                format_func=lambda x: x.strftime('%Y-%m-%d'),
                help="Date of the 3rd party valuation"
            )
        with col2:
            benchmark_value = st.number_input(
                "3rd Party Valuation ($M)",
                min_value=0.0,
                value=initial_value,
                step=10.0,
                help="Independent valuation estimate"
            )
        with col3:
            benchmark_source = st.text_input(
                "Source/Firm",
                value="Independent Appraiser",
                help="Who provided this valuation"
            )
        
        benchmark_data = {
            'date': benchmark_date,
            'value': benchmark_value,
            'source': benchmark_source
        }
    
    
    st.markdown("<hr style='border: 1px solid #000000; margin: 30px 0;'>", unsafe_allow_html=True)
    
    # ========================================================================
    # STEP 3: SECTOR & COMPARABLES (matching standard app)
    # ========================================================================
    
    st.markdown("<h2 style='color: #000000;'>3. Sector & Comparables Selection</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        sector = st.selectbox("Select Sector", list(SECTORS.keys()), index=0)
    with col2:
        comp_type = st.selectbox(
            "Comparable Type",
            ["Stock Prices", "Market Capitalization", "Indices", "Mixed"],
            index=0,
            help="Stock Prices: Easiest to obtain | Market Cap: Scale-matched | Indices: Broad exposure"
        )
    
    # Select Comparables
    available_comps = SECTORS[sector]["companies"]
    available_indices = SECTORS[sector]["indices"]
    
    # Comparable selection guidance
    with st.expander("🔍 Need help selecting comparables?", expanded=False):
        st.markdown("""
        ### How to Choose Good Comparables
        
        **Option 1: Similar Business Model**
        - Same industry vertical (e.g., SaaS, Biotech, Fintech)
        - Similar revenue model and growth stage
        - Comparable scale of operations
        
        **Option 2: Size-Matched**
        - Market cap within 0.5x - 2x of your company's estimated value
        - Similar operational scale
        
        **Option 3: Geographic/Market**
        - Operate in same markets
        - Similar customer base and competitive dynamics
        
        **Red Flags to Avoid:**
        - Companies in completely different sectors
        - Very different business models (B2B vs B2C)
        - Extreme size differences (micro-cap vs mega-cap)
        """)
        
        st.markdown("### 🔎 Search for Custom Comparables")
        search_term = st.text_input("Enter company ticker", placeholder="e.g., CRM, SNOW, DDOG", key="comp_search")
        
        if search_term:
            try:
                ticker_obj = yf.Ticker(search_term.upper())
                info = ticker_obj.info
                
                if info and 'longName' in info:
                    st.markdown(f"""
                    **{info.get('longName', 'Unknown')}** ({search_term.upper()})
                    - **Sector:** {info.get('sector', 'Unknown')}
                    - **Industry:** {info.get('industry', 'Unknown')}
                    - **Market Cap:** ${info.get('marketCap', 0)/1e9:.2f}B
                    - **Country:** {info.get('country', 'Unknown')}
                    """)
                    st.info(f"💡 Add {search_term.upper()} to the selection below if it's a good match")
                else:
                    st.warning("Could not find detailed info. Ticker may be invalid.")
            except Exception as e:
                st.warning(f"Could not find ticker: {str(e)}")
    
    selected_comps = []
    selected_indices = []
    custom_tickers = []
    
    if comp_type in ["Stock Prices", "Market Capitalization"]:
        selected_comps = st.multiselect(
            "Select Companies",
            available_comps,
            default=available_comps[:3],
            help="Choose 2-5 companies for best results"
        )
        
        # Allow custom tickers
        custom_input = st.text_input(
            "Add Custom Tickers (comma-separated)",
            placeholder="SNOW,DDOG,NET",
            help="Add additional tickers not in the predefined list"
        )
        if custom_input:
            custom_tickers = [t.strip().upper() for t in custom_input.split(',') if t.strip()]
    
    elif comp_type == "Indices":
        selected_indices = st.multiselect(
            "Select Indices",
            available_indices,
            default=available_indices[:2],
            help="Choose 1-3 indices for sector exposure"
        )
    else:  # Mixed
        col1, col2 = st.columns(2)
        with col1:
            selected_comps = st.multiselect(
                "Select Companies",
                available_comps,
                default=available_comps[:2]
            )
        with col2:
            selected_indices = st.multiselect(
                "Select Indices",
                available_indices,
                default=available_indices[:1]
            )
    
    all_tickers = selected_comps + selected_indices + custom_tickers
    
    if all_tickers:
        st.success(f"✅ {len(all_tickers)} comparables selected: {', '.join(all_tickers)}")
    else:
        st.warning("⚠️ Please select at least one comparable or index")
    
    
    st.markdown("<hr style='border: 1px solid #000000; margin: 30px 0;'>", unsafe_allow_html=True)
    
    # ========================================================================
    # STEP 4: MODEL PARAMETERS (matching standard app - NO SLIDERS)
    # ========================================================================
    
    st.markdown("<h2 style='color: #000000;'>4. Model Parameters</h2>", unsafe_allow_html=True)
    
    with st.expander("Configure Model Parameters", expanded=True):
        st.markdown("<h3 style='color: #000000;'>Uncertainty Parameters</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gp_marks_std_1 = st.number_input(
                "GP Mark 1 Std Dev",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.05,
                help="Uncertainty in first GP valuation source"
            )
        with col2:
            gp_marks_std_2 = st.number_input(
                "GP Mark 2 Std Dev",
                min_value=0.05,
                max_value=0.50,
                value=0.20,
                step=0.05,
                help="Uncertainty in second GP valuation source"
            )
        with col3:
            private_volatility = st.number_input(
                "Private Company Volatility",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.05,
                help="Expected volatility of private company"
            )
        
        st.markdown("<h3 style='color: #000000; margin-top: 20px;'>Advanced Options</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            comps_std = st.number_input(
                "Comparables Observation Std",
                min_value=0.001,
                max_value=0.05,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="Observation noise for public comparables"
            )
        with col2:
            dyn_tune = st.selectbox(
                "Dynamic Tuning",
                ["none", "EM"],
                index=1,
                help="EM algorithm learns optimal covariance structure"
            )
        with col3:
            obs_potential = st.selectbox(
                "Observation Potential",
                ["laplacian", "gaussian"],
                index=0,
                help="Laplacian is more robust to outliers"
            )
    
    
    st.markdown("<hr style='border: 1px solid #000000; margin: 30px 0;'>", unsafe_allow_html=True)
    
    # Main content check
    if not all_tickers:
        st.markdown("""
        <div style='background-color: #FFF3CD; padding: 15px; border-radius: 8px; border-left: 3px solid #000000; color: #000000;'>
            Please select at least one comparable or index above to continue.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # ========================================================================
    # STEP 5: CORRELATION SPECIFICATION (matching standard app)
    # ========================================================================
    
    st.markdown("<h2 style='color: #000000;'>5. Correlation Specification</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #FFF3CD; padding: 15px; border-radius: 8px; border-left: 3px solid #000000; color: #000000; margin-bottom: 20px;'>
        <strong>⚠️ Critical Parameter:</strong> Correlation estimates significantly impact model accuracy. 
        Wrong correlations can mislead the valuation estimates.
    </div>
    """, unsafe_allow_html=True)
    
    correlation_method = st.selectbox(
        "Correlation Estimation Method",
        ["Auto (from GP marks)", "Manual Override", "Equal Weights"],
        index=0,
        help="Auto uses GP marks as proxy. Manual allows you to specify correlations. Equal weights treats all comparables equally."
    )
    
    manual_correlations = {}
    if correlation_method == "Manual Override":
        st.markdown("<h3 style='color: #000000;'>Specify Correlations to Private Company</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #666666;'>Enter correlation values between 0.0 and 1.0 for each comparable:</p>", unsafe_allow_html=True)
        
        cols = st.columns(min(3, len(all_tickers)))
        for idx, ticker in enumerate(all_tickers):
            with cols[idx % 3]:
                manual_correlations[ticker] = st.number_input(
                    f"{ticker}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.65,
                    step=0.05,
                    help=f"Correlation between private company and {ticker}"
                )
    
    # Run button (centered and prominent)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_analysis = st.button("Run Valuation Analysis", type="primary", use_container_width=True)
    
    if run_analysis:
        with st.spinner("Fetching market data..."):
            # Fetch comparable data
            data_type_param = "market_cap" if comp_type == "Market Capitalization" else "price"
            comps_data = fetch_stock_data(all_tickers, start_date, end_date, data_type_param)
            
            if comps_data.empty:
                st.error("Could not fetch comparable data. Please try different tickers or date range.")
                st.stop()
            
            # Resample to quarterly
            comps_quarterly = resample_to_quarterly(comps_data, method='last')
            
            # Generate GP marks if not provided
            if gp_marks is None:
                gp_marks = generate_synthetic_gp_marks(
                    comps_quarterly.index,
                    base_value=initial_value,
                    num_marks=num_gp_marks
                )
            else:
                # Align GP marks with comparable dates
                common_dates = gp_marks.index.intersection(comps_quarterly.index)
                if len(common_dates) < len(gp_marks):
                    st.warning(f"⚠️ Only {len(common_dates)}/{len(gp_marks)} GP mark dates have comparable data")
                gp_marks = gp_marks.loc[common_dates]
            
            # Prepare cashflows (assume zero for simplicity)
            cashflows = pd.Series(0.0, index=comps_quarterly.index)
            
            st.markdown(f"""
            <div style='background-color: #D4EDDA; padding: 15px; border-radius: 8px; border-left: 3px solid #000000; color: #000000;'>
                Fetched data for {len(all_tickers)} comparables over {len(comps_quarterly)} quarters
            </div>
            """, unsafe_allow_html=True)
        
        # Display data overview
        st.markdown("<h2 style='color: #000000; margin-top: 40px;'>Data Overview & Validation</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Time Periods", len(comps_quarterly))
        with col2:
            st.metric("Comparables", len(all_tickers))
        with col3:
            st.metric("GP Mark Sources", num_gp_marks)
        with col4:
            gp_coverage = gp_marks.notna().sum().sum() / (len(gp_marks) * num_gp_marks) * 100
            st.metric("GP Coverage", f"{gp_coverage:.0f}%")
        
        # Show GP marks
        with st.expander("View GP Marks (Synthetic)", expanded=False):
            st.dataframe(gp_marks.style.format("{:.2f}"), use_container_width=True)
        
        # Show comparables data
        with st.expander("View Comparables Data", expanded=False):
            display_comps = comps_quarterly.copy()
            if comp_type == "Market Capitalization":
                display_comps = display_comps / 1e9  # Convert to billions
                st.markdown("*Values in billions*")
            st.dataframe(display_comps.style.format("{:.2f}"), use_container_width=True)
        
        # Prepare estimation inputs
        with st.spinner("Preparing model inputs..."):
            # Calculate correlations using average of GP marks as proxy
            private_proxy = gp_marks.mean(axis=1).dropna()
            
            # Align dates for correlation calculation
            common_dates = private_proxy.index.intersection(comps_quarterly.index)
            
            if correlation_method == "Manual Override":
                correlations = pd.Series(manual_correlations)
                raw_correlations = correlations.copy()
            elif correlation_method == "Equal Weights":
                correlations = pd.Series({ticker: 0.65 for ticker in all_tickers})
                raw_correlations = correlations.copy()
            else:  # Auto
                correlations, raw_correlations = calculate_correlations(
                    private_proxy.loc[common_dates],
                    comps_quarterly.loc[common_dates]
                )
            
            # Build parameters
            valuations_std = {f"GP_Mark_{i+1}": gp_marks_std_1 if i == 0 else gp_marks_std_2 
                            for i in range(num_gp_marks)}
            
            estimation_data = EstimationData(
                valuations=gp_marks,
                cashflows=cashflows,
                comparables=comps_quarterly
            )
            
            params = EstimationLogscaleParameters(
                valuations_std=pd.Series(valuations_std),
                volatility=private_volatility,
                correlations_to_private=correlations,
                idiosyncratic_growth=0.0
            )
            
            inputs = EstimationInputs(data=estimation_data, parameters=params)
        
        # Correlation validation
        st.markdown("<h3 style='color: #000000; margin-top: 30px;'>Correlation Validation</h3>", unsafe_allow_html=True)
        
        corr_df = pd.DataFrame({
            'Comparable': correlations.index,
            'Raw Correlation': raw_correlations.values.round(3),
            'Used in Model': correlations.values.round(3),
            'Status': ['✓ Good' if 0.4 <= c <= 0.9 else '⚠️ Weak' if c < 0.4 else '⚠️ Very High' 
                      for c in raw_correlations.values]
        })
        
        st.dataframe(
            corr_df.style.applymap(
                lambda x: 'background-color: #D4EDDA' if '✓' in str(x) else 'background-color: #FFF3CD' if '⚠️' in str(x) else '',
                subset=['Status']
            ),
            use_container_width=True,
            hide_index=True
        )
        
        avg_corr = correlations.mean()
        if avg_corr < 0.4:
            st.warning(f"⚠️ Average correlation is low ({avg_corr:.2f})")
        elif avg_corr > 0.85:
            st.info(f"ℹ️ Very high correlations ({avg_corr:.2f})")
        else:
            st.success(f"✓ Correlations look reasonable (avg: {avg_corr:.2f})")
        
        # Run models
        st.markdown("<h2 style='color: #000000; margin-top: 40px;'>Valuation Results</h2>", unsafe_allow_html=True)
        
        with st.spinner("Running RollForward baseline..."):
            baseline = RollForward()(inputs)
        
        with st.spinner("Running NAVcasting model (Production Version)..."):
            estimator = NAVcasting(
                dyn_tune=dyn_tune,
                comps_std=comps_std,
                obs_potential=obs_potential,
                dyn_potential="gaussian"
            )
            result = estimator(inputs)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            latest_nav = result.nav_estimate['central'].iloc[-1]
            latest_baseline = baseline.nav_estimate['central'].iloc[-1]
            uncertainty_pct = (result.nav_estimate['upper'].iloc[-1] - result.nav_estimate['lower'].iloc[-1]) / latest_nav * 50
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Latest NAVcasting", f"${latest_nav:.2f}M")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Latest RollForward", f"${latest_baseline:.2f}M")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                diff_pct = (latest_nav - latest_baseline) / latest_baseline * 100
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Difference", f"{diff_pct:+.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Uncertainty", f"±{uncertainty_pct:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 3rd party comparison
            if use_benchmark and benchmark_data is not None:
                st.markdown("<h3 style='color: #000000; margin-top: 20px;'>3rd Party Benchmark Comparison</h3>", unsafe_allow_html=True)
                
                model_val_at_benchmark = result.nav_estimate.loc[benchmark_data['date'], 'central']
                diff_from_benchmark = (model_val_at_benchmark - benchmark_data['value']) / benchmark_data['value'] * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("3rd Party Valuation", f"${benchmark_data['value']:.2f}M")
                with col2:
                    st.metric("NAVcasting at Same Date", f"${model_val_at_benchmark:.2f}M")
                with col3:
                    st.metric("Difference", f"{diff_from_benchmark:+.1f}%", 
                             delta_color="off" if abs(diff_from_benchmark) < 10 else "normal")
                
                if abs(diff_from_benchmark) < 10:
                    st.success("✅ Model closely aligns with 3rd party valuation")
                elif abs(diff_from_benchmark) < 20:
                    st.info("ℹ️ Model shows moderate difference from 3rd party valuation")
                else:
                    st.warning("⚠️ Significant difference from 3rd party valuation - review assumptions")
            
            # Detailed results table
            st.markdown("<h3 style='color: #000000; margin-top: 30px;'>Time Series Results</h3>", unsafe_allow_html=True)
            
            results_df = pd.DataFrame({
                'Date': result.nav_estimate.index.strftime('%Y-%m-%d'),
                'NAVcasting': result.nav_estimate['central'],
                'NAV_Lower': result.nav_estimate['lower'],
                'NAV_Upper': result.nav_estimate['upper'],
                'RollForward': baseline.nav_estimate['central'],
                'GP_Mark_1': gp_marks['GP_Mark_1'],
                'GP_Mark_2': gp_marks['GP_Mark_2'] if 'GP_Mark_2' in gp_marks.columns else np.nan
            })
            
            st.dataframe(
                results_df.style.format({
                    'NAVcasting': '{:.2f}',
                    'NAV_Lower': '{:.2f}',
                    'NAV_Upper': '{:.2f}',
                    'RollForward': '{:.2f}',
                    'GP_Mark_1': '{:.2f}',
                    'GP_Mark_2': '{:.2f}'
                }, na_rep='—'),
                use_container_width=True
            )
            
            # ====================================================================
            # VISUALIZATIONS
            # ====================================================================
            
            st.markdown("<h2 style='color: #000000; margin-top: 40px;'>Visualizations</h2>", unsafe_allow_html=True)
            
            # Plot 1: Valuation Comparison
            fig1 = go.Figure()
            
            # Add uncertainty band
            fig1.add_trace(go.Scatter(
                x=result.nav_estimate.index,
                y=result.nav_estimate['upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig1.add_trace(go.Scatter(
                x=result.nav_estimate.index,
                y=result.nav_estimate['lower'],
                mode='lines',
                name='NAVcasting ±1σ',
                fill='tonexty',
                fillcolor='rgba(0, 100, 255, 0.2)',
                line=dict(width=0),
                showlegend=True
            ))
            
            # Add GP marks
            for col in gp_marks.columns:
                fig1.add_trace(go.Scatter(
                    x=gp_marks.index,
                    y=gp_marks[col],
                    mode='markers',
                    name=col,
                    marker=dict(size=10, symbol='circle')
                ))
            
            # Add baseline
            fig1.add_trace(go.Scatter(
                x=baseline.nav_estimate.index,
                y=baseline.nav_estimate['central'],
                mode='lines',
                name='RollForward',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            # Add NAVcasting
            fig1.add_trace(go.Scatter(
                x=result.nav_estimate.index,
                y=result.nav_estimate['central'],
                mode='lines',
                name='NAVcasting',
                line=dict(color='blue', width=3)
            ))
            
            fig1.update_layout(
                title=f"{company_name} - Valuation Comparison",
                xaxis_title="Date",
                yaxis_title="Valuation ($M)",
                hovermode='x unified',
                height=500,
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#FFFFFF',
                font=dict(color='#000000'),
                xaxis=dict(gridcolor='#E0E0E0', showline=True, linecolor='#000000'),
                yaxis=dict(gridcolor='#E0E0E0', showline=True, linecolor='#000000')
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Plot 2: Comparables Performance
            fig2 = go.Figure()
            
            # Normalize comparables to start at 100
            comps_normalized = (comps_quarterly / comps_quarterly.iloc[0]) * 100
            
            for col in comps_normalized.columns:
                fig2.add_trace(go.Scatter(
                    x=comps_normalized.index,
                    y=comps_normalized[col],
                    mode='lines',
                    name=col,
                    opacity=0.7
                ))
            
            fig2.update_layout(
                title="Comparables Performance (Indexed to 100)",
                xaxis_title="Date",
                yaxis_title="Index Value",
                hovermode='x unified',
                height=400,
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#FFFFFF',
                font=dict(color='#000000'),
                xaxis=dict(gridcolor='#E0E0E0', showline=True, linecolor='#000000'),
                yaxis=dict(gridcolor='#E0E0E0', showline=True, linecolor='#000000')
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Plot 3: Correlation Analysis
            st.markdown("<h3 style='color: #000000; margin-top: 40px;'>Correlation Analysis</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("<h4 style='color: #000000;'>Correlation Summary</h4>", unsafe_allow_html=True)
                for ticker in correlations.index:
                    st.metric(
                        ticker,
                        f"{correlations[ticker]:.2f}",
                        delta=f"Raw: {raw_correlations[ticker]:.2f}" if correlation_method == "Auto (Calculate from GP Marks)" else None
                    )
            
            with col2:
                st.markdown("<h4 style='color: #000000;'>Visual Validation: GP Marks vs Comparables</h4>", unsafe_allow_html=True)
                
                # Create scatter plot
                fig_corr = go.Figure()
                
                gp_returns = np.log(private_proxy).diff().dropna()
                comp_returns = np.log(comps_quarterly.loc[common_dates]).diff().dropna()
                
                for ticker in correlations.index:
                    if ticker in comp_returns.columns:
                        aligned_gp = gp_returns.loc[comp_returns.index]
                        fig_corr.add_trace(go.Scatter(
                            x=comp_returns[ticker],
                            y=aligned_gp,
                            mode='markers',
                            name=f"{ticker} (ρ={raw_correlations[ticker]:.2f})",
                            marker=dict(size=8, opacity=0.6)
                        ))
                
                fig_corr.update_layout(
                    title="Log Returns: GP Marks vs Comparables",
                    xaxis_title="Comparable Log Returns",
                    yaxis_title="GP Marks Log Returns (Proxy)",
                    height=400,
                    paper_bgcolor='#FFFFFF',
                    plot_bgcolor='#FFFFFF',
                    font=dict(color='#000000'),
                    xaxis=dict(gridcolor='#E0E0E0', showline=True, linecolor='#000000', zeroline=True, zerolinecolor='#CCCCCC'),
                    yaxis=dict(gridcolor='#E0E0E0', showline=True, linecolor='#000000', zeroline=True, zerolinecolor='#CCCCCC'),
                    showlegend=True
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.markdown("""
                <p style='color: #666666; font-size: 14px;'>
                <strong>How to interpret:</strong> Points closer to a diagonal line indicate stronger correlation.
                </p>
                """, unsafe_allow_html=True)
            
            # ====================================================================
            # EXPORT & SCENARIO ANALYSIS
            # ====================================================================
            
            st.markdown("<h2 style='color: #000000; margin-top: 40px;'>Export & Analysis</h2>", unsafe_allow_html=True)
            
            # Scenario Analysis
            with st.expander("🎯 Scenario Analysis & Sensitivity Testing", expanded=False):
                st.markdown("### Test Model Sensitivity to Key Assumptions")
                
                scenario = st.selectbox(
                    "Select Scenario",
                    [
                        "Base Case (Current Settings)",
                        "Conservative (Higher Uncertainty)",
                        "Optimistic (Lower Uncertainty)",
                        "Correlation Stress Test",
                    ],
                    key="scenario_selector"
                )
                
                if st.button("Run Scenario Analysis", key="run_scenario"):
                    with st.spinner("Running scenario analysis..."):
                        scenario_results = {'Base Case': result.nav_estimate['central'].iloc[-1]}
                        
                        if scenario == "Conservative (Higher Uncertainty)":
                            cons_params = EstimationLogscaleParameters(
                                valuations_std=params.valuations_std * 1.5,
                                volatility=params.volatility * 1.3,
                                correlations_to_private=params.correlations_to_private,
                                idiosyncratic_growth=params.idiosyncratic_growth
                            )
                            cons_inputs = EstimationInputs(data=estimation_data, parameters=cons_params)
                            cons_result = estimator(cons_inputs)
                            
                            scenario_results['Conservative'] = cons_result.nav_estimate['central'].iloc[-1]
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Base Case", f"${scenario_results['Base Case']:.2f}M")
                            col2.metric("Conservative", f"${scenario_results['Conservative']:.2f}M",
                                      delta=f"{((scenario_results['Conservative']/scenario_results['Base Case'])-1)*100:.1f}%")
                        
                        elif scenario == "Optimistic (Lower Uncertainty)":
                            opt_params = EstimationLogscaleParameters(
                                valuations_std=params.valuations_std * 0.7,
                                volatility=params.volatility * 0.8,
                                correlations_to_private=params.correlations_to_private,
                                idiosyncratic_growth=params.idiosyncratic_growth
                            )
                            opt_inputs = EstimationInputs(data=estimation_data, parameters=opt_params)
                            opt_result = estimator(opt_inputs)
                            
                            scenario_results['Optimistic'] = opt_result.nav_estimate['central'].iloc[-1]
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Base Case", f"${scenario_results['Base Case']:.2f}M")
                            col2.metric("Optimistic", f"${scenario_results['Optimistic']:.2f}M",
                                      delta=f"{((scenario_results['Optimistic']/scenario_results['Base Case'])-1)*100:.1f}%")
                        
                        elif scenario == "Correlation Stress Test":
                            st.markdown("**Testing sensitivity to correlation assumptions**")
                            
                            stress_levels = [0.3, 0.5, 0.7, 0.9]
                            stress_results = {}
                            
                            for stress in stress_levels:
                                stress_corr = pd.Series({ticker: stress for ticker in correlations.index})
                                stress_params = EstimationLogscaleParameters(
                                    valuations_std=params.valuations_std,
                                    volatility=params.volatility,
                                    correlations_to_private=stress_corr,
                                    idiosyncratic_growth=params.idiosyncratic_growth
                                )
                                stress_inputs = EstimationInputs(data=estimation_data, parameters=stress_params)
                                stress_result = estimator(stress_inputs)
                                stress_results[f"Corr={stress}"] = stress_result.nav_estimate['central'].iloc[-1]
                            
                            stress_df = pd.DataFrame.from_dict(stress_results, orient='index', columns=['Latest Valuation ($M)'])
                            st.dataframe(stress_df.style.format("{:.2f}"), use_container_width=True)
                            
                            val_range = max(stress_results.values()) - min(stress_results.values())
                            sensitivity_pct = (val_range / scenario_results['Base Case']) * 100
                            
                            if sensitivity_pct > 30:
                                st.warning(f"⚠️ High sensitivity to correlations ({sensitivity_pct:.0f}% range)")
                            else:
                                st.success(f"✓ Moderate sensitivity to correlations ({sensitivity_pct:.0f}% range)")
            
            # Export section
            st.markdown("### Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_df = pd.DataFrame({
                    'Date': result.nav_estimate.index.strftime('%Y-%m-%d'),
                    'NAVcasting_Central': result.nav_estimate['central'],
                    'NAVcasting_Lower': result.nav_estimate['lower'],
                    'NAVcasting_Upper': result.nav_estimate['upper'],
                    'RollForward_Central': baseline.nav_estimate['central'],
                    **{col: gp_marks[col] for col in gp_marks.columns}
                })
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Data (CSV)",
                    data=csv,
                    file_name=f"navcasting_results_{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                uncertainty_pct = (result.nav_estimate['upper'].iloc[-1] - result.nav_estimate['lower'].iloc[-1]) / result.nav_estimate['central'].iloc[-1] * 50
                corr_list = '\n'.join([f"- {ticker}: {correlations[ticker]:.3f}" for ticker in correlations.index])
                
                report_md = f"""# NAVcasting Valuation Report - PRODUCTION VERSION
## {company_name}
**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Executive Summary
- **Latest Valuation:** ${result.nav_estimate['central'].iloc[-1]:.2f}M ± {uncertainty_pct:.1f}%
- **Method:** NAVcasting with {len(all_tickers)} comparables
- **Backend:** alog_nav.py (production-grade with comprehensive documentation)

### Key Results
| Metric | Value |
|--------|-------|
| NAVcasting Latest | ${result.nav_estimate['central'].iloc[-1]:.2f}M |
| RollForward Latest | ${baseline.nav_estimate['central'].iloc[-1]:.2f}M |
| Difference | {((result.nav_estimate['central'].iloc[-1] - baseline.nav_estimate['central'].iloc[-1])/baseline.nav_estimate['central'].iloc[-1]*100):+.1f}% |

### Comparables Used
{', '.join(all_tickers)}

### Correlations
{corr_list}

**Generated by:** NAVcasting Production App (alog_nav.py backend)
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                
                st.download_button(
                    label="📝 Download Report (MD)",
                    data=report_md,
                    file_name=f"navcasting_report_{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            
            # Model info
            with st.expander("ℹ️ Model Information", expanded=False):
                st.markdown(f"""
                <div style='color: #000000;'>
                <h3 style='color: #000000;'>Production Model Configuration</h3>
                
                <p><strong>Backend:</strong> alog_nav.py with comprehensive documentation</p>
                <p><strong>Data Type:</strong> {comp_type}</p>
                
                <p><strong>Model Parameters:</strong></p>
                <ul>
                <li>GP Mark 1 Std Dev: {gp_marks_std_1}</li>
                <li>GP Mark 2 Std Dev: {gp_marks_std_2}</li>
                <li>Private Volatility: {private_volatility}</li>
                <li>Comparables Std: {comps_std}</li>
                <li>Dynamic Tuning: {dyn_tune}</li>
                <li>Observation Potential: {obs_potential}</li>
                </ul>
                
                <p><strong>Algorithm Details:</strong></p>
                <ul>
                <li>RollForward: Baseline with precision-weighted averaging</li>
                <li>NAVcasting: Bayesian MAP estimation with EM tuning</li>
                <li>Loss Functions: Configurable (Gaussian/Laplacian)</li>
                <li>Uncertainty: Quantified via inverse Hessian</li>
                </ul>
                
                <p><strong>Documentation:</strong></p>
                <p>This production version uses alog_nav.py which includes extensive docstrings 
                explaining the mathematical framework, algorithm variants, and implementation details.
                See the backend code for comprehensive documentation.</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
