import os
import streamlit as st
import pandas as pd
import sys
import importlib
from datetime import datetime
import json
import time
import warnings
from typing import Dict, Any

# Import the research agent modules
from research_agent import ResearchAgent
from memory import EnhancedMemory
from tools import LLMTool, NewsSearchTool, SemanticFilterTool, RelevanceEvaluatorTool
from citation_tracker import CitationTracker
from answer_evaluation import AnswerEvaluator, AnswerRegenerator, EvaluationResult

# Import UI components
from ui_components import (
    PacManProgressVisualizer, StreamingOutputCapture, format_text_for_display,
    stream_text_progressively, extract_citations_from_answer, display_quality_badge,
    update_live_stats
)

# Configure Streamlit page
st.set_page_config(
    page_title="AI News Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*_embedding_bag.*")

# Custom CSS (keeping the essential styles)
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    .citation-box {
        background-color: #f0f2f6; padding: 12px; border-radius: 8px;
        margin: 8px 0; border-left: 4px solid #1f77b4;
    }
    .citation-link { color: #1f77b4; text-decoration: none; font-weight: 500; }
    .citation-link:hover { text-decoration: underline; color: #0d47a1; }
    .quality-badge {
        padding: 8px 16px; border-radius: 20px; font-weight: bold;
        display: inline-block; margin: 5px 5px 5px 0;
    }
    .excellent { background-color: #4CAF50; color: white; }
    .good { background-color: #FFC107; color: black; }
    .fair { background-color: #FF9800; color: white; }
    .poor { background-color: #F44336; color: white; }
    .streaming-console {
        background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace;
        padding: 15px; border-radius: 8px; max-height: 400px; overflow-y: auto;
        font-size: 12px; margin: 10px 0; border: 2px solid #00ff00;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
    }
    .answer-section {
        background-color: #f8f9fa; padding: 20px; border-radius: 10px;
        margin: 15px 0; border: 1px solid #e9ecef;
    }
    .answer-text, .streaming-text {
        font-size: 14px !important; line-height: 1.7 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        white-space: pre-line !important; word-wrap: break-word !important;
        margin-bottom: 16px !important;
    }
    .status-indicator {
        padding: 8px 16px; border-radius: 20px; font-weight: bold;
        display: inline-block; margin: 5px 0;
    }
    .status-running { background-color: #ffc107; color: black; animation: pulse 2s infinite; }
    .status-complete { background-color: #28a745; color: white; }
    .status-error { background-color: #dc3545; color: white; }
    .streaming-cursor { animation: blink 1s infinite; font-weight: bold; color: #007bff; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0; } 100% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_research' not in st.session_state:
    st.session_state.current_research = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'console_output' not in st.session_state:
    st.session_state.console_output = []
if 'is_researching' not in st.session_state:
    st.session_state.is_researching = False
if 'streaming_output' not in st.session_state:
    st.session_state.streaming_output = []
if 'live_stats' not in st.session_state:
    st.session_state.live_stats = {
        'iterations': 0, 'articles_found': 0, 'current_step': 'Ready',
        'elapsed_time': 0, 'status': 'ready', 'planning_failures': 0
    }
if 'pacman_high_score' not in st.session_state:
    st.session_state.pacman_high_score = 0

def initialize_agent():
    """Initialize the research agent with tools and memory"""
    with st.spinner("Initializing research agent..."):
        tools = [
            LLMTool(), NewsSearchTool(), SemanticFilterTool(), RelevanceEvaluatorTool()
        ]
        memory = EnhancedMemory()
        agent = ResearchAgent("HierarchicalResearcher", tools, memory)
        
        if hasattr(agent, 'clear_cache'):
            agent.clear_cache()
        if hasattr(agent.memory, 'clear'):
            agent.memory.clear()
        
        return agent

def run_streaming_research(question):
    """Run research with real-time streaming output and PAC-Man visualization"""
    
    # Reset streaming state
    st.session_state.streaming_output = []
    st.session_state.live_stats = {
        'iterations': 0, 'articles_found': 0, 'current_step': 'Initializing...',
        'elapsed_time': 0, 'status': 'running', 'planning_failures': 0
    }
    
    # Create streaming interface
    st.subheader("🖥️ Live Research Stream")
    
    # Create placeholders
    stats_placeholder = st.empty()
    pacman_placeholder = st.empty()
    console_placeholder = st.empty()
    
    # Initialize PAC-Man visualizer
    pacman_viz = PacManProgressVisualizer()
    update_live_stats(stats_placeholder)
    
    # Stats callback with PAC-Man update
    def stats_callback():
        pacman_viz = PacManProgressVisualizer()
        total_phases = len(pacman_viz.phases)

        if st.session_state.live_stats.get('status') == 'complete':
            current_phase = total_phases
            phase_progress = {}
        else:
            current_phase = 0
            phase_progress = {}
            
            if 'agent' in st.session_state and st.session_state.agent:
                agent = st.session_state.agent
                if hasattr(agent, 'planner') and hasattr(agent.planner, 'subgoals'):
                    phase_progress = agent.planner.subgoals

                    if isinstance(phase_progress, dict):
                        phase_names = list(phase_progress.keys())
                        for idx, key in enumerate(phase_names):
                            phase_data = phase_progress.get(key, {})
                            if not isinstance(phase_data, dict):
                                continue

                            if phase_data.get("complete", False):
                                current_phase = idx + 1
                                continue

                            steps = phase_data.get("steps", [])
                            if isinstance(steps, list):
                                for step in steps:
                                    if isinstance(step, dict) and step.get("done", False):
                                        current_phase = idx
                                        break
                                if current_phase == idx:
                                    break

        failures = st.session_state.live_stats.get('planning_failures', 0)
        high_score = st.session_state.pacman_high_score
        is_running = (st.session_state.live_stats.get('status') == 'running')

        try:
            pacman_html = pacman_viz.generate_pacman_html(
                current_phase=current_phase, phase_progress=phase_progress,
                is_researching=is_running, failures=failures, high_score=high_score
            )
            pacman_placeholder.empty()
            with pacman_placeholder.container():
                st.markdown(pacman_html, unsafe_allow_html=True)
        except Exception as e:
            pacman_placeholder.empty()
            with pacman_placeholder.container():
                st.info(f"🎮 Research Progress: Phase {current_phase}/{total_phases} · Score: {high_score}")

        update_live_stats(stats_placeholder)
    
    # Create fresh agent
    st.session_state.live_stats['current_step'] = 'Creating fresh agent instance...'
    stats_callback()
    
    if 'agent' in st.session_state:
        del st.session_state.agent
    
    agent = initialize_agent()
    st.session_state.agent = agent
    start_time = time.time()
    
    progress_bar = st.progress(0)
    
    # Run research with streaming
    try:
        st.session_state.live_stats['current_step'] = 'Starting research...'
        st.session_state.live_stats['status'] = 'running'
        stats_callback()
        
        with StreamingOutputCapture(console_placeholder, stats_callback):
            print(f"🚀 Starting hierarchical research with streaming...")
            print(f"📝 Research question: {question}")
            print(f"🔧 Agent: {agent.name} with {len(agent.tools)} tools")
            progress_bar.progress(0.1)
            
            print("🔧 Initializing research tools and memory...")
            progress_bar.progress(0.2)
            
            print("🎯 Setting high-quality research parameters...")
            
            # Execute research
            articles_df, results = agent.execute_research_with_chaining(question)
            
            print("✅ Research completed successfully!")
            progress_bar.progress(1.0)
            
            total_time = time.time() - start_time
            
            # Final stats update
            st.session_state.live_stats['elapsed_time'] = total_time
            st.session_state.live_stats['status'] = 'complete'
            st.session_state.live_stats['current_step'] = 'Research completed!'
            st.session_state.live_stats['planning_failures'] = results.get('planning_failures', 0)
            stats_callback()
            
            # Update high score
            current_score = pacman_viz._calculate_score(results.get('progress_report', {}))
            if current_score > st.session_state.pacman_high_score:
                st.session_state.pacman_high_score = current_score
                st.session_state.last_bonus_score = current_score
                st.balloons()
            
            # Store results
            research_result = {
                'question': question, 'timestamp': datetime.now(),
                'articles_df': articles_df, 'results': results,
                'total_time': total_time, 'console_log': list(st.session_state.streaming_output),
                'streaming': True, 'agent_name': agent.name, 'fresh_agent': True
            }
            
            return research_result
            
    except Exception as e:
        st.session_state.live_stats['status'] = 'error'
        st.session_state.live_stats['current_step'] = f'Error: {str(e)[:40]}...'
        stats_callback()
        progress_bar.progress(1.0)
        st.error(f"Research failed: {str(e)}")
        return None

def display_research_results(research_result):
    """Display research results with citation analysis"""
    if not research_result:
        return
    
    results = research_result['results']
    articles_df = research_result['articles_df']
    answer_text = results.get('answer', 'No answer generated')
    
    # Extract citations
    cited_ids = extract_citations_from_answer(answer_text)
    
    # Verify citations against articles
    verified_citations = []
    missing_citations = []
    
    for citation_id in cited_ids:
        found = False
        if not articles_df.empty:
            for col in ['news_id', 'id', 'citation_id', 'url_id']:
                if col in articles_df.columns:
                    mask1 = articles_df[col].astype(str).str.contains(citation_id, na=False)
                    mask2 = articles_df[col].astype(str).str.contains(f"news_{citation_id}", na=False)
                    if mask1.any() or mask2.any():
                        verified_citations.append(citation_id)
                        found = True
                        break
        if not found:
            missing_citations.append(citation_id)
    
    # Calculate enhanced quality score
    original_score = results.get('evaluation_stats', {}).get('final_score', 0.0)
    citation_accuracy = len(verified_citations) / len(cited_ids) if cited_ids else 1.0
    hallucination_penalty = min(len(missing_citations) * 0.05, 0.2)
    
    enhanced_score = max(0.0, min(1.0, 
        original_score * 0.6 + citation_accuracy * 0.35 + 0.05 - hallucination_penalty
    ))
    
    if enhanced_score >= 0.8:
        verdict = "Excellent"
    elif enhanced_score >= 0.65:
        verdict = "Good"
    elif enhanced_score >= 0.45:
        verdict = "Fair"
    else:
        verdict = "Poor"
    
    # Display answer section
    st.markdown('<div class="answer-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📝 Research Answer")
    with col2:
        st.markdown(display_quality_badge(verdict), unsafe_allow_html=True)
        st.caption(f"Enhanced Score: {enhanced_score:.2f}/1.0 ({enhanced_score*100:.0f}%)")
        st.caption(f"Original Score: {original_score:.2f}/1.0")
    
    # Display the answer
    if research_result.get('streaming', False):
        answer_placeholder = st.empty()
        streaming_method = getattr(st.session_state, 'streaming_speed', 'fast_words')
        stream_text_progressively(answer_text, answer_placeholder, method=streaming_method)
    else:
        formatted_answer = format_text_for_display(answer_text)
        st.markdown(f'<div class="answer-text">{formatted_answer.replace("\n\n", "<br><br>")}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Citations section
    st.subheader("🔍 Citation Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Citations", len(cited_ids))
    with col2:
        st.metric("✅ Verified", len(verified_citations))
    with col3:
        st.metric("❌ Missing", len(missing_citations))
    with col4:
        st.metric("Accuracy", f"{citation_accuracy:.1%}")
    
    if cited_ids:
        citation_text = ", ".join([f"[news_{cid}]" for cid in cited_ids])
        st.code(citation_text, language="text")

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("Skywalker - AI News Research Agent")
    st.markdown("*Hierarchical research system with PAC-Man visualization and enhanced citation verification*")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        streaming_mode = st.checkbox("🌊 Enable Streaming Mode", value=True)
        
        if streaming_mode:
            streaming_speed = st.selectbox(
                "Streaming Speed:",
                options=["instant", "fast_words", "chunks", "sentences"],
                index=1
            )
            st.session_state.streaming_speed = streaming_speed
        
        if st.button("🔄 Reload Modules"):
            modules = ['research_agent', 'memory', 'tools', 'ui_components']
            for module in modules:
                if module in sys.modules:
                    try:
                        importlib.reload(sys.modules[module])
                    except Exception as e:
                        st.error(f"Failed to reload {module}: {str(e)}")
        
        if st.button("🗑️ Clear Cache & Reset"):
            for key in ['current_research', 'console_output', 'streaming_output']:
                if key in st.session_state:
                    st.session_state[key] = [] if 'output' in key else None
            st.session_state.is_researching = False
            st.success("Cache cleared!")
            st.rerun()
        
        st.subheader("🎮 PAC-Man High Score")
        st.metric("🏆 High Score", st.session_state.pacman_high_score)
        
        # Research history
        st.subheader("Research History")
        if st.session_state.research_history:
            for i, research in enumerate(reversed(st.session_state.research_history[-5:])):
                timestamp = research['timestamp'].strftime("%H:%M:%S")
                question_preview = research['question'][:25] + "..." if len(research['question']) > 25 else research['question']
                if st.button(f"📄 {timestamp}: {question_preview}", key=f"hist_{i}"):
                    st.session_state.current_research = research
                    st.rerun()
        else:
            st.info("No research history yet")
    
    # Main content
    st.header("Research Question")
    
    with st.expander("📌 Example Questions"):
        st.markdown("""
        - How is Harvard's endowment situation?
        - What are the latest developments in AI regulation?
        - What is the current state of climate change policy?
        """)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Enter your research question:",
            placeholder="e.g., How is Harvard's endowment situation?",
            key="question_input"
        )
    
    with col2:
        st.write(""); st.write("")
        button_text = "🌊 Stream Research" if streaming_mode else "🔍 Research"
        research_button = st.button(
            button_text, type="primary", use_container_width=True,
            disabled=st.session_state.is_researching
        )
    
    # Research execution
    if research_button and question and not st.session_state.is_researching:
        st.session_state.is_researching = True
        st.divider()
        
        if 'agent' in st.session_state:
            del st.session_state.agent
        
        if streaming_mode:
            research_result = run_streaming_research(question)
        else:
            # Non-streaming fallback (simplified)
            with st.spinner("Conducting research..."):
                agent = initialize_agent()
                start_time = time.time()
                
                try:
                    articles_df, results = agent.execute_research_with_chaining(question)
                    total_time = time.time() - start_time
                    
                    research_result = {
                        'question': question, 'timestamp': datetime.now(),
                        'articles_df': articles_df, 'results': results,
                        'total_time': total_time, 'streaming': False
                    }
                except Exception as e:
                    st.error(f"Research failed: {str(e)}")
                    research_result = None
        
        # Handle results
        if research_result:
            st.session_state.current_research = research_result
            st.session_state.research_history.append(research_result)
            
            quality_score = research_result['results'].get('evaluation_stats', {}).get('final_score', 0)
            st.success(f"✅ Research completed in {research_result['total_time']:.1f} seconds! Quality: {quality_score:.1%}")
            
            if quality_score >= 0.8:
                st.balloons()
        
        st.session_state.is_researching = False
    
    # Display results
    if st.session_state.current_research and not st.session_state.is_researching:
        st.divider()
        st.header("📊 Research Results")
        display_research_results(st.session_state.current_research)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    Powered by Hierarchical Research Agent v2.2 | PAC-Man Progress Visualization | Enhanced Citation Verification
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()