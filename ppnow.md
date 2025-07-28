import streamlit as st

def apply_theme():
    """Apply the Netflix-style theme and CSS to the app"""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    body {
        font-family: 'Roboto', sans-serif;
        font-size: 0.8125rem;
        margin: 0;
        padding: 0;
        background-color: #141414;
        color: #ffffff;
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
        background-color: #ffffff;
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
        color: #ffffff;
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
        color: #ffffff;
    }

    .nav-item.active {
        background-color: rgba(229, 9, 20, 0.1);
        font-weight: 500;
        color: #E50914;
        border-bottom: 2px solid #E50914;
    }

    /* Blue button style for secondary navigation */
    .blue-button {
        background-color: transparent;
        color: #E50914;
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
        color: #ffffff;
        transform: none;
        box-shadow: none;
    }

    .app-title {
        position: relative;
        font-size: 18px;
        font-weight: 500;
        color: #000000;
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
        color: #000000;
        font-family: 'Roboto', sans-serif;
        z-index: 1;
    }
    
    .r-symbol {
        font-size: 8px;
        vertical-align: super;
        margin-left: 2px;
        position: relative;
        bottom: 8px;
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
        color: #ffffff;
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
        color: #999;
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
        background-color: #141414;
        color: #ffffff;
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
        background: linear-gradient(45deg, #e50914, #b81111);
        color: #ffffff;
        border: none;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3);
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
        background: linear-gradient(45deg, #ff1a2b, #e50914);
        transform: translateY(-2px);
        box-shadow: 
            0 6px 20px rgba(229, 9, 20, 0.4),
            0 0 30px rgba(229, 9, 20, 0.3);
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover::after {
        opacity: 1;
    }
    
    .product-card {
        background: linear-gradient(145deg, #181818, #1a1a1a);
        border-radius: 8px;
        padding: 20px;
        margin: 10px;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(229, 9, 20, 0.2);
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
        transform: scale(1.05) translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5), 0 0 20px rgba(229, 9, 20, 0.3);
        border-color: rgba(229, 9, 20, 0.6);
    }
    
    .product-card:hover::before {
        opacity: 1;
    }
    
    .product-card h3 {
        color: white;
        margin-bottom: 10px;
    }
    
    .product-card h4 {
        color: #e50914;
        font-size: 12px;
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .product-card p {
        color: #ccc;
    }
    
    .product-card strong {
        color: white;
    }
    
    .client-card {
        background: linear-gradient(145deg, #1a1a1a, #2a2a2a);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        color: white;
        border: 2px solid rgba(229, 9, 20, 0.3);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .client-card:hover {
        transform: scale(1.03) translateY(-3px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5), 0 0 20px rgba(229, 9, 20, 0.4);
        border-color: rgba(229, 9, 20, 0.6);
    }
    
    .hero-section {
        background: 
            linear-gradient(135deg, rgba(0,0,0,0.9) 0%, rgba(229, 9, 20, 0.3) 30%, rgba(229, 9, 20, 0.2) 70%, rgba(0,0,0,0.9) 100%),
            linear-gradient(45deg, #141414 0%, #1a1a1a 50%, #141414 100%);
        background-size: 400% 400%;
        animation: futuristicGradient 20s ease infinite;
        padding: 100px 20px;
        text-align: center;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
        color: white;
        margin-top: -50px;
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
        color: white;
        position: relative;
        z-index: 2;
        text-shadow: 
            0 0 10px rgba(229, 9, 20, 0.8),
            0 0 20px rgba(229, 9, 20, 0.6),
            0 0 30px rgba(229, 9, 20, 0.4);
        animation: text-glow 4s ease-in-out infinite alternate;
    }
    
    .hero-section p {
        color: white;
        position: relative;
        z-index: 2;
        text-shadow: 
            0 0 5px rgba(229, 9, 20, 0.6),
            0 0 10px rgba(229, 9, 20, 0.4);
        animation: text-pulse 3s ease-in-out infinite alternate;
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
        text-align: center;
        padding: 40px;
        background: linear-gradient(145deg, #222, #2a2a2a);
        border-radius: 8px;
        border: 1px solid rgba(229, 9, 20, 0.3);
        color: white;
    }
    
    .home-option-card h3 {
        color: white;
        margin-bottom: 15px;
    }
    
    .home-option-card p {
        color: #ccc;
        margin: 0;
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
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        transition: all 0.3s ease;
        background: linear-gradient(45deg, rgba(229, 9, 20, 0.2), rgba(229, 9, 20, 0.4));
    }
    
    .attribute-pill:hover {
        transform: scale(1.1);
        box-shadow: 0 2px 8px rgba(229, 9, 20, 0.4);
    }
    
    .traffic-light {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: inline-block;
        margin: 0 2px;
        animation: pulse 2s ease-in-out infinite;
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
        background: radial-gradient(circle, #4CAF50, #2E7D32);
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
    }
    .yellow-light { 
        background: radial-gradient(circle, #FFC107, #F57C00);
        box-shadow: 0 0 10px rgba(255, 193, 7, 0.5);
    }
    .red-light { 
        background: radial-gradient(circle, #f44336, #c62828);
        box-shadow: 0 0 10px rgba(244, 67, 54, 0.5);
    }
    
    .match-explanation {
        background: linear-gradient(145deg, #222, #2a2a2a);
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        border-left: 3px solid #e50914;
        animation: slideIn 0.5s ease-out;
        color: white;
    }
    
    .match-explanation strong {
        color: white;
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
        <h1 style="font-size: 48px; margin-bottom: 20px;">Matchmaker</h1>
        <p style="font-size: 24px; margin-bottom: 40px;">Find the perfect match between investment products and client preferences</p>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render the application footer with dark theme"""
    st.markdown('''
    <div style="position: fixed; bottom: 0; left: 0; right: 0; padding: 20px; background-color: #f8f9fa; border-top: 1px solid #dee2e6; z-index: 1001;">
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <p style="color: #000000; font-size: 16px; margin: 0; text-align: center;">
                © 2025 MASS Solutions <span style="color: #E50914;">❤️</span> Made By PAG
            </p>
        </div>
    </div>
    <div style="height: 80px;"></div>
    ''', unsafe_allow_html=True)
