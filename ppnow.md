# app.py

try:
    from load_env import auto_load
    auto_load()
except ImportError:
    print("Note: load_env.py not found. Using system environment variables.")

import streamlit as st
from matchmaker import DataManager, MatchingEngine, ExcelHandler

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Matchmaker",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import UI components
from ui import theme

# Apply theme globally with force application - this ensures it's applied on every page render
# This should be called immediately after st.set_page_config
theme.force_theme_application() if hasattr(theme, 'force_theme_application') else theme.apply_theme()

# Import pages after theme is applied
from pages.funds import show_funds_page
from pages.clients import show_clients_page
from pages.whatif import show_whatif_page
from pages.matching import show_matching_page
from pages.excel_editor import show_excel_editor_page

# Initialize session state with library components
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
    st.session_state.data_manager.create_sample_data()

if 'matching_engine' not in st.session_state:
    st.session_state.matching_engine = MatchingEngine()

if 'excel_handler' not in st.session_state:
    st.session_state.excel_handler = ExcelHandler()

# Initialize navigation state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_fund' not in st.session_state:
    st.session_state.selected_fund = None
if 'selected_client' not in st.session_state:
    st.session_state.selected_client = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = None

def home_page():
    """Display the home page"""
    # Theme is already applied globally, but we still need navbar and background
    theme.render_navbar(active_page='home')
    theme.render_animated_background()
    theme.render_hero_section()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # First row - primary options
        col_clients, col_funds = st.columns(2)
        
        with col_clients:
            st.markdown("""
            <div class="home-option-card">
                <h3>Browse Clients</h3>
                <p>View client preferences and discover suitable products</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start with Clients", key="btn_clients", use_container_width=True):
                st.session_state.page = 'clients'
                st.session_state.view_mode = 'clients'
                st.rerun()
        
        with col_funds:
            st.markdown("""
            <div class="home-option-card">
                <h3>Browse Funds</h3>
                <p>Explore our investment products and find matching clients</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start with Funds", key="btn_funds", use_container_width=True):
                st.session_state.page = 'funds'
                st.session_state.view_mode = 'funds'
                st.rerun()
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Second row - additional options
        col_whatif, col_excel = st.columns(2)
        
        with col_whatif:
            st.markdown("""
            <div class="home-option-card">
                <h3>What-If Clients</h3>
                <p>Create hypothetical scenarios and test matching</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Create What-If", key="btn_whatif", use_container_width=True):
                st.session_state.page = 'whatif'
                st.rerun()
        
        with col_excel:
            st.markdown("""
            <div class="home-option-card">
                <h3>Excel Manager</h3>
                <p>Edit data and matching rules via Excel Editor</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Excel Editor", key="btn_excel", use_container_width=True):
                st.session_state.page = 'excel'
                st.rerun()
    
    # Add footer at the end of the home page
    theme.render_footer()

# Main app routing
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'funds':
    show_funds_page()
elif st.session_state.page == 'clients':
    show_clients_page()
elif st.session_state.page == 'whatif':
    show_whatif_page()
elif st.session_state.page == 'matching':
    show_matching_page()
elif st.session_state.page == 'excel':
    show_excel_editor_page()
else:
    home_page()
