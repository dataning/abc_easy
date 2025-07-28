import streamlit as st

def apply_theme():
    """Apply the Netflix-style theme and CSS to the app with dark mode prevention"""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    
    /* Force light mode and prevent system dark mode interference */
    :root {
        color-scheme: light !important;
    }
    
    html {
        color-scheme: light !important;
    }
    
    /* Override Streamlit's theme variables */
    :root {
        --primary-color: #E50914 !important;
        --background-color: #141414 !important;
        --secondary-background-color: #1a1a1a !important;
        --text-color: #ffffff !important;
        --font: 'Roboto', sans-serif !important;
    }
    
    /* Force background colors on all Streamlit containers */
    .stApp {
        background-color: #141414 !important;
        color: #ffffff !important;
    }
    
    .main .block-container {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #141414 !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Override any Streamlit dark mode classes */
    .st-emotion-cache-1y4p8pa {
        background-color: #141414 !important;
    }
    
    .st-emotion-cache-z5fcl4 {
        background-color: #141414 !important;
    }
    
    /* Force text colors */
    * {
        color-scheme: light !important;
    }
    
    body {
        font-family: 'Roboto', sans-serif !important;
        font-size: 0.8125rem !important;
        margin: 0 !important;
        padding: 0 !important;
        background-color: #141414 !important;
        color: #ffffff !important;
    }
    
    /* Ensure all text elements use white color */
    p, span, div, h1, h2, h3, h4, h5, h6, label {
        color: #ffffff !important;
    }
    
    /* Force input fields to have proper styling */
    input, textarea, select {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 1px solid rgba(229, 9, 20, 0.3) !important;
    }
    
    /* Override Streamlit's sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    a {
        text-decoration: none !important;
        color: #ffffff !important;
    }

    /* Footer always light background and font color globally */
    div[style*="position: fixed"][style*="bottom: 0"] * {
        color: #000000 !important;
    }
    div[style*="position: fixed"][style*="bottom: 0"] {
        background-color: #f8f9fa !important;
        border-top: 1px solid #dee2e6 !important;
    }
    
    /* Top navbar styles - UPDATED TO DARK THEME */
    .app-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 60px;
        background-color: #ffffff !important;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding: 0 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        z-index: 1000;
        border-bottom: 1px solid rgba(229, 9, 20, 0.3);
    }

    .nav-items {
        display: flex;
        height: 100%;
    }

    .nav-item {
        height: 40px;
        display: flex;
        align-items: center;
        padding: 0 15px;
        font-size: 14px;
        color: #ffffff !important;
        text-decoration: none;
        position: relative;
        font-family: 'Roboto', sans-serif;
        transition: all 0.3s ease;
        cursor: pointer;
        border: none;
        background: transparent;
    }

    .nav-item:hover {
        background-color: rgba(229, 9, 20, 0.2);
        border-bottom: 2px solid #E50914;
        color: #ffffff !important;
    }

    .nav-item.active {
        background-color: rgba(229, 9, 20, 0.1);
        font-weight: 500;
        color: #E50914 !important;
        border-bottom: 2px solid #E50914;
    }

    /* Blue button style for secondary navigation */
    .blue-button {
        background-color: transparent;
        color: #E50914 !important;
        border: 1px solid #E50914;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 400;
        padding: 8px 16px;
        height: 36px;
        transition: all 0.3s ease;
        cursor: pointer;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .blue-button:hover {
        background-color: #E50914;
        color: #ffffff !important;
        transform: none;
        box-shadow: none;
    }

    .app-title {
        position: relative;
        font-size: 18px;
        font-weight: 500;
        color: #000000 !important;
        font-family: 'Roboto', sans-serif;
        margin-left: 50px;
        margin-top: 0;
        text-transform: uppercase;
        z-index: 2;
    }

    .app-name {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        font-size: 32px;
        font-weight: 800;
        color: #000000 !important;
        font-family: 'Roboto', sans-serif;
        z-index: 1;
    }
    
    .r-symbol {
        font-size: 8px;
        vertical-align: super;
        margin-left: 2px;
        position: relative;
        bottom: 8px;
        color: #000000 !important;
    }

    .nav-actions {
        display: flex;
        align-items: center;
        padding-right: 20px;
    }

    .dropdown-arrow {
        margin-left: 5px;
        font-size: 10px;
    }

    .options-button {
        background: none;
        border: none;
        color: #ffffff !important;
        font-size: 14px;
        cursor: pointer;
        display: flex;
        align-items: center;
        height: 40px;
        padding: 0 15px;
    }

    .options-button:hover {
        background-color: rgba(229, 9, 20, 0.2);
        border-bottom: 2px solid #E50914;
    }
    
    .top-right-buttons {
        position: absolute;
        right: 20px;
        top: 0;
        height: 60px;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .icon-button {
        display: flex;
        align-items: center;
        font-size: 14px;
        color: #999 !important;
        text-decoration: none;
        cursor: pointer;
        font-family: 'Roboto', sans-serif;
        transition: all 0.3s ease;
    }
    
    .icon-button svg {
        margin-right: 5px;
    }

    /* Animated Background */
    .main {
        background-color: #141414 !important;
        color: #ffffff !important;
        position: relative;
        min-height: 100vh;
        margin-top: 60px;
    }
    
    /* Full page animated background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: 
            radial-gradient(circle at 20% 80%, rgba(229, 9, 20, 0.4) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(229, 9, 20, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 60% 70%, rgba(229, 9, 20, 0.2) 0%, transparent 40%);
        animation: backgroundMove 25s ease-in-out infinite;
        z-index: -10;
        pointer-events: none;
    }
    
    /* Additional moving background layer */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: 
            radial-gradient(ellipse at 10% 30%, rgba(229, 9, 20, 0.15) 0%, transparent 60%),
            radial-gradient(ellipse at 90% 70%, rgba(255, 255, 255, 0.05) 0%, transparent 60%);
        animation: backgroundMove2 30s ease-in-out infinite reverse;
        z-index: -9;
        pointer-events: none;
    }
    
    @keyframes backgroundMove {
        0%, 100% {
            transform: translateX(0px) translateY(0px) scale(1);
        }
        25% {
            transform: translateX(-50px) translateY(-30px) scale(1.1);
        }
        50% {
            transform: translateX(30px) translateY(50px) scale(0.9);
        }
        75% {
            transform: translateX(20px) translateY(-40px) scale(1.05);
        }
    }
    
    @keyframes backgroundMove2 {
        0%, 100% {
            transform: translateX(0px) translateY(0px) scale(1) rotate(0deg);
        }
        33% {
            transform: translateX(40px) translateY(-60px) scale(1.2) rotate(120deg);
        }
        66% {
            transform: translateX(-30px) translateY(40px) scale(0.8) rotate(240deg);
        }
    }
    
    /* Floating particles */
    .floating-particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: -8;
        overflow: hidden;
    }
    
    .particle {
        position: absolute;
        border-radius: 50%;
        animation: float 15s infinite linear;
    }
    
    .particle:nth-child(odd) {
        background: radial-gradient(circle, rgba(229, 9, 20, 0.8), rgba(229, 9, 20, 0.2));
        box-shadow: 
            0 0 10px rgba(229, 9, 20, 0.8),
            0 0 20px rgba(229, 9, 20, 0.6),
            0 0 30px rgba(229, 9, 20, 0.4);
    }
    
    .particle:nth-child(even) {
        background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.2));
        box-shadow: 
            0 0 10px rgba(255, 255, 255, 0.6),
            0 0 20px rgba(255, 255, 255, 0.4),
            0 0 30px rgba(255, 255, 255, 0.2);
    }
    
    .particle:nth-child(3n) {
        background: radial-gradient(circle, rgba(229, 9, 20, 0.6), rgba(229, 9, 20, 0.1));
        box-shadow: 
            0 0 10px rgba(229, 9, 20, 0.6),
            0 0 20px rgba(229, 9, 20, 0.4),
            0 0 30px rgba(229, 9, 20, 0.2);
    }
    
    .particle:nth-child(1) {
        width: 6px;
        height: 6px;
        left: 10%;
        animation-delay: 0s;
        animation-duration: 20s;
    }
    
    .particle:nth-child(2) {
        width: 4px;
        height: 4px;
        left: 30%;
        animation-delay: 5s;
        animation-duration: 25s;
    }
    
    .particle:nth-child(3) {
        width: 8px;
        height: 8px;
        left: 60%;
        animation-delay: 10s;
        animation-duration: 18s;
    }
    
    .particle:nth-child(4) {
        width: 5px;
        height: 5px;
        left: 80%;
        animation-delay: 15s;
        animation-duration: 22s;
    }
    
    .particle:nth-child(5) {
        width: 3px;
        height: 3px;
        left: 50%;
        animation-delay: 8s;
        animation-duration: 28s;
    }
    
    .particle:nth-child(6) {
        width: 7px;
        height: 7px;
        left: 25%;
        animation-delay: 12s;
        animation-duration: 24s;
    }
    
    @keyframes float {
        0% {
            transform: translateY(100vh) translateX(0px) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translateY(-10vh) translateX(50px) rotate(360deg);
            opacity: 0;
        }
    }
    
    /* Full screen grid pattern */
    .grid-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-image: 
            linear-gradient(rgba(229, 9, 20, 0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(229, 9, 20, 0.08) 1px, transparent 1px);
        background-size: 60px 60px;
        animation: gridMove 40s linear infinite;
        z-index: -7;
        pointer-events: none;
    }
    
    @keyframes gridMove {
        0% {
            transform: translate(0, 0);
        }
        100% {
            transform: translate(60px, 60px);
        }
    }
    
    /* Add subtle moving lines */
    .moving-lines {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -6;
        pointer-events: none;
        overflow: hidden;
    }
    
    .moving-lines::before {
        content: '';
        position: absolute;
        top: 20%;
        left: -10%;
        width: 120%;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(229, 9, 20, 0.3), transparent);
        animation: moveLine1 25s linear infinite;
    }
    
    .moving-lines::after {
        content: '';
        position: absolute;
        top: 70%;
        left: -10%;
        width: 120%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: moveLine2 35s linear infinite reverse;
    }
    
    @keyframes moveLine1 {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(100%);
        }
    }
    
    @keyframes moveLine2 {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(100%);
        }
    }
    
    /* Netflix-style buttons with red theme */
    .stButton>button {
        background: linear-gradient(45deg, #e50914, #b81111) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 12px 30px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3) !important;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.2), 
            transparent);
        transition: left 0.5s ease;
    }
    
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(45deg, #ff1a2b, #e50914) !important;
        transform: translateY(-2px) !important;
        box-shadow: 
            0 6px 20px rgba(229, 9, 20, 0.4),
            0 0 30px rgba(229, 9, 20, 0.3) !important;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover::after {
        opacity: 1;
    }
    
    .product-card {
        background: linear-gradient(145deg, #181818, #1a1a1a) !important;
        border-radius: 8px !important;
        padding: 20px !important;
        margin: 10px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        border: 1px solid rgba(229, 9, 20, 0.2) !important;
    }
    
    .product-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(229, 9, 20, 0.1), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .product-card:hover {
        transform: scale(1.05) translateY(-5px) !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5), 0 0 20px rgba(229, 9, 20, 0.3) !important;
        border-color: rgba(229, 9, 20, 0.6) !important;
    }
    
    .product-card:hover::before {
        opacity: 1;
    }
    
    .product-card h3 {
        color: white !important;
        margin-bottom: 10px !important;
    }
    
    .product-card h4 {
        color: #e50914 !important;
        font-size: 12px !important;
        margin: 0 0 8px 0 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .product-card p {
        color: #ccc !important;
    }
    
    .product-card strong {
        color: white !important;
    }
    
    .client-card {
        background: linear-gradient(145deg, #1a1a1a, #2a2a2a) !important;
        padding: 15px !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        color: white !important;
        border: 2px solid rgba(229, 9, 20, 0.3) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .client-card:hover {
        transform: scale(1.03) translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5), 0 0 20px rgba(229, 9, 20, 0.4) !important;
        border-color: rgba(229, 9, 20, 0.6) !important;
    }
    
    .hero-section {
        background: 
            linear-gradient(135deg, rgba(0,0,0,0.9) 0%, rgba(229, 9, 20, 0.3) 30%, rgba(229, 9, 20, 0.2) 70%, rgba(0,0,0,0.9) 100%),
            linear-gradient(45deg, #141414 0%, #1a1a1a 50%, #141414 100%) !important;
        background-size: 400% 400% !important;
        animation: futuristicGradient 20s ease infinite !important;
        padding: 100px 20px !important;
        text-align: center !important;
        margin-bottom: 40px !important;
        position: relative !important;
        overflow: hidden !important;
        color: white !important;
        margin-top: -50px !important;
    }
    
    @keyframes futuristicGradient {
        0%, 100% { background-position: 0% 50%; }
        25% { background-position: 100% 0%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            repeating-linear-gradient(
                0deg,
                transparent 0px,
                rgba(229, 9, 20, 0.4) 1px,
                transparent 2px,
                transparent 50px
            ),
            repeating-linear-gradient(
                90deg,
                transparent 0px,
                rgba(229, 9, 20, 0.3) 1px,
                transparent 2px,
                transparent 80px
            ),
            radial-gradient(circle at 20% 50%, rgba(229, 9, 20, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 50%, rgba(229, 9, 20, 0.2) 0%, transparent 50%);
        animation: highway-matrix 8s linear infinite;
        opacity: 0.6;
        z-index: 1;
        pointer-events: none;
    }
    
    @keyframes highway-matrix {
        0% { 
            transform: translateX(-50px) translateY(-100px) scale(1); 
            filter: hue-rotate(0deg);
        }
        50% { 
            transform: translateX(0px) translateY(-50px) scale(1.05); 
            filter: hue-rotate(180deg);
        }
        100% { 
            transform: translateX(50px) translateY(0px) scale(1); 
            filter: hue-rotate(360deg);
        }
    }
    
    .hero-section::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            linear-gradient(90deg, 
                transparent 0%, 
                rgba(229, 9, 20, 0.8) 5%, 
                transparent 10%, 
                transparent 90%, 
                rgba(229, 9, 20, 0.8) 95%, 
                transparent 100%
            ),
            linear-gradient(45deg, 
                transparent 0%, 
                rgba(229, 9, 20, 0.4) 25%, 
                transparent 50%, 
                rgba(255, 255, 255, 0.3) 75%, 
                transparent 100%
            );
        animation: hyperspace-sweep 3s ease-in-out infinite;
        z-index: 1;
        pointer-events: none;
    }
    
    @keyframes hyperspace-sweep {
        0% { 
            transform: translateX(-150%) scaleX(0.3);
            opacity: 0;
        }
        30% {
            transform: translateX(-50%) scaleX(1);
            opacity: 1;
        }
        70% {
            transform: translateX(50%) scaleX(1);
            opacity: 1;
        }
        100% { 
            transform: translateX(150%) scaleX(0.3);
            opacity: 0;
        }
    }
    
    .hero-section h1 {
        color: white !important;
        position: relative !important;
        z-index: 2 !important;
        text-shadow: 
            0 0 10px rgba(229, 9, 20, 0.8),
            0 0 20px rgba(229, 9, 20, 0.6),
            0 0 30px rgba(229, 9, 20, 0.4) !important;
        animation: text-glow 4s ease-in-out infinite alternate !important;
    }
    
    .hero-section p {
        color: white !important;
        position: relative !important;
        z-index: 2 !important;
        text-shadow: 
            0 0 5px rgba(229, 9, 20, 0.6),
            0 0 10px rgba(229, 9, 20, 0.4) !important;
        animation: text-pulse 3s ease-in-out infinite alternate !important;
    }
    
    @keyframes text-glow {
        0% { 
            text-shadow: 
                0 0 10px rgba(229, 9, 20, 0.8),
                0 0 20px rgba(229, 9, 20, 0.6),
                0 0 30px rgba(229, 9, 20, 0.4);
        }
        100% { 
            text-shadow: 
                0 0 20px rgba(229, 9, 20, 1),
                0 0 30px rgba(229, 9, 20, 0.8),
                0 0 40px rgba(229, 9, 20, 0.6);
        }
    }
    
    @keyframes text-pulse {
        0% { 
            text-shadow: 
                0 0 5px rgba(229, 9, 20, 0.6),
                0 0 10px rgba(229, 9, 20, 0.4);
        }
        100% { 
            text-shadow: 
                0 0 10px rgba(229, 9, 20, 0.8),
                0 0 15px rgba(229, 9, 20, 0.6),
                0 0 20px rgba(229, 9, 20, 0.4);
        }
    }
    
    .home-option-card {
        text-align: center !important;
        padding: 40px !important;
        background: linear-gradient(145deg, #222, #2a2a2a) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(229, 9, 20, 0.3) !important;
        color: white !important;
    }
    
    .home-option-card h3 {
        color: white !important;
        margin-bottom: 15px !important;
    }
    
    .home-option-card p {
        color: #ccc !important;
        margin: 0 !important;
    }
    
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    .hero-section::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 30% 20%, rgba(229, 9, 20, 0.4) 0%, transparent 50%),
            radial-gradient(circle at 70% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
        animation: heroParticles 25s ease-in-out infinite;
    }
    
    @keyframes heroParticles {
        0%, 100% {
            transform: scale(1) rotate(0deg);
            opacity: 0.6;
        }
        50% {
            transform: scale(1.2) rotate(180deg);
            opacity: 0.8;
        }
    }
    
    .attribute-pill {
        display: inline-block !important;
        padding: 4px 12px !important;
        margin: 2px !important;
        border-radius: 20px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(45deg, rgba(229, 9, 20, 0.2), rgba(229, 9, 20, 0.4)) !important;
        color: #ffffff !important;
    }
    
    .attribute-pill:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 2px 8px rgba(229, 9, 20, 0.4) !important;
    }
    
    .traffic-light {
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
        display: inline-block !important;
        margin: 0 2px !important;
        animation: pulse 2s ease-in-out infinite !important;
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.1);
            opacity: 0.8;
        }
    }
    
    .green-light { 
        background: radial-gradient(circle, #4CAF50, #2E7D32) !important;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.5) !important;
    }
    .yellow-light { 
        background: radial-gradient(circle, #FFC107, #F57C00) !important;
        box-shadow: 0 0 10px rgba(255, 193, 7, 0.5) !important;
    }
    .red-light { 
        background: radial-gradient(circle, #f44336, #c62828) !important;
        box-shadow: 0 0 10px rgba(244, 67, 54, 0.5) !important;
    }
    
    .match-explanation {
        background: linear-gradient(145deg, #222, #2a2a2a) !important;
        padding: 15px !important;
        border-radius: 8px !important;
        margin-top: 10px !important;
        border-left: 3px solid #e50914 !important;
        animation: slideIn 0.5s ease-out !important;
        color: white !important;
    }
    
    .match-explanation strong {
        color: white !important;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .stMarkdown, .stColumns {
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Override any system or browser dark mode preferences */
    @media (prefers-color-scheme: dark) {
        :root {
            color-scheme: light !important;
        }
        
        * {
            color-scheme: light !important;
        }
        
        body {
            background-color: #141414 !important;
            color: #ffffff !important;
        }
    }
    
    /* Force specific Streamlit components to use our theme */
    [data-baseweb="select"] {
        background-color: #2a2a2a !important;
    }
    
    [data-baseweb="input"] {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }
    
    /* Override Streamlit's emotion cache classes that might interfere */
    [class*="st-emotion-cache"] {
        color: inherit !important;
        background-color: transparent !important;
    }
    
    /* Ensure all Streamlit widgets follow the theme */
    .stSelectbox > div > div {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }
    
    /* Force override for any Streamlit theme detection */
    .streamlit-wide {
        background-color: #141414 !important;
    }
    
    .element-container {
        color: #ffffff !important;
    }
    
    /* Ensure metrics and other components use white text */
    [data-testid="metric-container"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cccccc !important;
    }
</style>
""", unsafe_allow_html=True)

def render_navbar(active_page='home'):
    """Render the navigation bar"""
    st.markdown(f"""
    <div class="app-header">
        <div class="app-title">MASS Matchmaking</div>
        <div class="app-name">Aladdin<span class="r-symbol">®</span></div>
        <div class="top-right-buttons"></div>
    </div>
    <div style="height: 60px;"></div>
    """, unsafe_allow_html=True)

def render_animated_background():
    """Render animated background elements that cover the entire app"""
    st.markdown("""
    <div class="grid-background"></div>
    <div class="moving-lines"></div>
    <div class="floating-particles">
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
    </div>
    """, unsafe_allow_html=True)

def render_hero_section():
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 48px; margin-bottom: 20px; color: white !important;">Matchmaker</h1>
        <p style="font-size: 24px; margin-bottom: 40px; color: white !important;">Find the perfect match between investment products and client preferences</p>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render the application footer with dark theme"""
    st.markdown('''
    <div style="position: fixed; bottom: 0; left: 0; right: 0; padding: 20px; background-color: #f8f9fa !important; border-top: 1px solid #dee2e6; z-index: 1001;">
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <p style="color: #000000 !important; font-size: 16px; margin: 0; text-align: center;">
                © 2025 MASS Solutions <span style="color: #E50914;">❤️</span> Made By PAG
            </p>
        </div>
    </div>
    <div style="height: 80px;"></div>
    ''', unsafe_allow_html=True)

# Additional helper function to force theme application
def force_theme_application():
    """Call this at the beginning of your app to ensure theme is applied"""
    # Set Streamlit config to force light theme
    st.set_page_config(
        page_title="MASS Matchmaking - Aladdin",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom theme immediately
    apply_theme()
    
    # Inject JavaScript to override any system preferences
    st.markdown("""
    <script>
        // Force disable dark mode at the document level
        document.documentElement.style.colorScheme = 'light';
        document.body.style.colorScheme = 'light';
        
        // Override any Streamlit theme detection
        if (window.localStorage) {
            window.localStorage.setItem('streamlit-theme', 'light');
        }
        
        // Force background color on body
        document.body.style.backgroundColor = '#141414';
        
        // Continuously check and force our theme
        setInterval(() => {
            const root = document.documentElement;
            const body = document.body;
            
            // Force color scheme
            root.style.colorScheme = 'light';
            body.style.colorScheme = 'light';
            
            // Force background colors
            body.style.backgroundColor = '#141414';
            
            // Find and update any Streamlit containers
            const appView = document.querySelector('[data-testid="stAppViewContainer"]');
            if (appView) {
                appView.style.backgroundColor = '#141414';
            }
            
            const stApp = document.querySelector('.stApp');
            if (stApp) {
                stApp.style.backgroundColor = '#141414';
            }
        }, 100);
    </script>
    """, unsafe_allow_html=True)
