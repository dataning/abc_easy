import streamlit as st
import pandas as pd
import sys
import time
import re
from typing import Dict, Any
from datetime import datetime

class PacManProgressVisualizer:
    """PAC-Man visualization for research progress"""
    
    def __init__(self):
        self.phases = [
            "🔍 Question Analysis",
            "🔍 Keyword Extraction", 
            "🔍 Article Retrieval",
            "🔍 Relevance Filtering",
            "🔍 Domain Diversification",
            "🔍 Answer Generation"
        ]
    
    def generate_pacman_html(self, current_phase: int, phase_progress: Dict[str, Any], 
                           is_researching: bool = True, failures: int = 0,
                           high_score: int = 0) -> str:
        """Generate PAC-Man visualization HTML"""
        
        current_score = self._calculate_score(phase_progress)
        total_phases = len(self.phases)
        progress_percent = (current_phase / total_phases) * 100 if total_phases > 0 else 0
        pacman_position = min(85, progress_percent * 0.85)
        
        # Generate dots
        dots_html = ""
        for i in range(0, 90, 10):
            dots_html += f'<span style="position:absolute; left:{i}%; top:45%; color:#FFD700; font-size:8px;">•</span>'
        
        # Generate ghosts for failures
        ghosts_html = ""
        ghost_emojis = ["👻", "👾", "🤖", "💀"]
        for i in range(min(failures, 4)):
            ghost_pos = 70 + (i * 5)
            if ghost_pos < pacman_position:
                ghosts_html += f'<span style="position:absolute; left:{ghost_pos}%; top:40%; font-size:20px;">{ghost_emojis[i]}</span>'
        
        # Phase status list
        phase_list_html = ""
        for idx, phase in enumerate(self.phases):
            if idx < current_phase:
                status, color = "✅", "#00FF00"
            elif idx == current_phase:
                status, color = "🟡", "#FFD700"
            else:
                status, color = "⚪", "#666666"
            
            phase_list_html += f'''<div style="margin:5px 0; padding:5px; background:rgba(0,0,0,0.3); border-left:3px solid {color}; border-radius:5px;">{status} {phase}</div>'''
        
        # Status message
        if not is_researching:
            if current_phase >= len(self.phases):
                status_msg = "🎉 RESEARCH COMPLETE! ALL PHASES CLEARED! 🎉"
                status_color = "#00FF00"
            else:
                status_msg = "⏸️ Research Paused"
                status_color = "#FFD700"
        else:
            if current_phase < len(self.phases):
                status_msg = f"🔄 {self.phases[current_phase]} IN PROGRESS..."
                status_color = "#FFD700"
            else:
                status_msg = "🎉 FINAL PHASE COMPLETE!"
                status_color = "#00FF00"
        
        # Build complete HTML
        html = f'''<div style="background: linear-gradient(135deg, #001122, #003366); border: 3px solid #FFD700; border-radius: 15px; padding: 20px; margin: 20px 0; min-height: 350px; font-family: 'Courier New', monospace; color: #FFD700; position: relative;">
<div style="text-align:center; margin-bottom:15px; font-size:18px; font-weight:bold;">🎮 PAC-MAN RESEARCH PROGRESS 🎮</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 15px; font-size: 14px; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 8px;">
<div>SCORE: <span style="color:#FFD700;">{current_score:,}</span></div>
<div>PHASE: <span style="color:#FFD700;">{current_phase + 1}/{len(self.phases)}</span></div>
<div>FAILURES: <span style="color:#FFD700;">{failures}</span></div>
<div>HIGH: <span style="color:#FFD700;">{high_score:,}</span></div>
</div>
<div style="text-align: center; margin: 15px 0; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; color: {status_color}; font-size: 16px; font-weight: bold;">{status_msg}</div>
<div style="position: relative; height: 60px; background: #000033; border: 2px solid #0080FF; border-radius: 30px; margin: 20px 0; overflow: hidden;">
{dots_html}
<div style="position: absolute; left: {pacman_position}%; top: 50%; transform: translateY(-50%); font-size: 30px; z-index: 10;">🟡</div>
{ghosts_html}
<div style="position: absolute; right: 15px; top: 50%; transform: translateY(-50%); font-size: 20px;">🍒</div>
</div>
<div style="background: #000000; height: 10px; border-radius: 5px; margin: 15px 0; overflow: hidden;">
<div style="background: linear-gradient(90deg, #FFD700, #FFA500); height: 100%; width: {progress_percent}%; transition: width 0.5s ease;"></div>
</div>
<div style="margin-top: 20px;">
<div style="font-weight:bold; margin-bottom:10px;">🎯 RESEARCH PHASES:</div>
{phase_list_html}
</div>
</div>'''
        
        return html
    
    def _calculate_score(self, phase_progress: Dict[str, Any]) -> int:
        """Calculate game score"""
        score = 0
        try:
            if isinstance(phase_progress, dict):
                for phase_name, progress in phase_progress.items():
                    if isinstance(progress, dict):
                        steps = progress.get("steps", [])
                        if isinstance(steps, list):
                            completed = sum(1 for step in steps if isinstance(step, dict) and step.get("done", False))
                            score += completed * 100
                        
                        if progress.get("complete", False):
                            score += 500
        except:
            score = 100
        
        return max(score, 100)

class StreamingOutputCapture:
    """Enhanced output capture with real-time streaming"""
    
    def __init__(self, console_placeholder, stats_callback=None):
        self.console_placeholder = console_placeholder
        self.stats_callback = stats_callback
        self.output_buffer = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.start_time = time.time()
        
    def write(self, text):
        self.original_stdout.write(text)
        
        if text.strip():
            timestamp = f"[{time.time() - self.start_time:.1f}s]"
            formatted_text = f"{timestamp} {text}"
            
            self.output_buffer.append(formatted_text)
            st.session_state.streaming_output.append(formatted_text)
            
            self._parse_for_stats(text)
            self._update_display()
            
            # Keep buffer manageable
            if len(self.output_buffer) > 100:
                self.output_buffer = self.output_buffer[-80:]
            if len(st.session_state.streaming_output) > 100:
                st.session_state.streaming_output = st.session_state.streaming_output[-80:]
    
    def _parse_for_stats(self, text):
        """Extract statistics from output text"""
        current_time = time.time() - self.start_time
        st.session_state.live_stats['elapsed_time'] = current_time
        text_lower = text.lower()

        # Parse iterations
        if "iteration" in text_lower:
            match = re.search(r'iteration\s+(\d+)', text_lower)
            if match:
                st.session_state.live_stats['iterations'] = int(match.group(1))

        # Parse article counts
        if "articles" in text_lower:
            match = re.search(r'(\d+)\s+articles', text_lower)
            if match:
                st.session_state.live_stats['articles_found'] = int(match.group(1))

        # Parse failures
        if "planning failures" in text_lower:
            match = re.search(r'planning failures:\s*(\d+)', text_lower)
            if match:
                st.session_state.live_stats['planning_failures'] = int(match.group(1))
        elif "failed" in text_lower or "failure" in text_lower:
            current_failures = st.session_state.live_stats.get('planning_failures', 0)
            st.session_state.live_stats['planning_failures'] = current_failures + 1

        # Parse current step
        if any(emoji in text for emoji in ["🔍", "📊", "🤔", "✅", "⚠️", "🔄"]):
            clean_step = text.strip()[:60]
            if clean_step:
                st.session_state.live_stats['current_step'] = clean_step

        # Status updates
        if "research completed successfully" in text_lower:
            st.session_state.live_stats['status'] = 'complete'
        elif "error" in text_lower or "failed" in text_lower:
            st.session_state.live_stats['status'] = 'error'
        else:
            st.session_state.live_stats['status'] = 'running'
    
    def _update_display(self):
        """Update the Streamlit display with latest output"""
        if self.output_buffer:
            recent_output = self.output_buffer[-20:]
            output_text = '<br>'.join(recent_output)
            
            with self.console_placeholder.container():
                st.markdown(
                    f'<div class="streaming-console">{output_text}</div>',
                    unsafe_allow_html=True
                )
            
            if self.stats_callback:
                self.stats_callback()
    
    def flush(self):
        self.original_stdout.flush()
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

def format_text_for_display(text):
    """Format text with paragraph breaks for better readability"""
    if not text:
        return ""
    
    text = text.strip()
    
    # Protect citations
    citation_placeholders = {}
    citation_counter = 0
    
    citation_pattern = r'\[news_[a-f0-9]{6,12}\]|\[[a-f0-9]{6,12}\]'
    citations = re.findall(citation_pattern, text)
    
    for citation in citations:
        placeholder = f"__CITATION_{citation_counter}__"
        citation_placeholders[placeholder] = citation
        text = text.replace(citation, placeholder, 1)
        citation_counter += 1
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    formatted_paragraphs = []
    current_paragraph = []
    sentence_count = 0
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        sentence = sentence.strip()
        current_paragraph.append(sentence)
        sentence_count += 1
        
        # Break logic
        should_break = False
        
        if sentence_count >= 2:
            transition_indicators = [
                'according to', 'furthermore', 'moreover', 'however', 'in addition',
                'for instance', 'this move', 'the issue', 'these reports', 'analysis',
                'from a', 'the situation', 'this indicates', 'in analyzing', 'recent',
                'this development', 'as reported', 'the trump', 'republicans',
                'despite', 'while the', 'as the', 'in conclusion', 'the future',
                'furthermore', 'the termination', 'the endowment'
            ]
            
            sentence_lower = sentence.lower()
            has_transition = any(sentence_lower.startswith(indicator) for indicator in transition_indicators)
            should_break = sentence_count >= 3 or (sentence_count >= 2 and has_transition)
        
        if i == len(sentences) - 1:
            should_break = True
        
        if should_break:
            paragraph_text = ' '.join(current_paragraph).strip()
            if paragraph_text:
                formatted_paragraphs.append(paragraph_text)
            
            current_paragraph = []
            sentence_count = 0
    
    result = '\n\n'.join(formatted_paragraphs)
    
    # Restore citations
    for placeholder, citation in citation_placeholders.items():
        result = result.replace(placeholder, citation)
    
    return result

def stream_text_progressively(text, placeholder, method="fast_words", delay=None):
    """Stream text with paragraph support"""
    
    formatted_text = format_text_for_display(text)
    
    base_style = '''style="
        font-size: 14px; 
        line-height: 1.7; 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        white-space: pre-line;
        word-wrap: break-word;
    "'''
    
    if method == "instant":
        placeholder.markdown(f'<div class="answer-text" {base_style}>{formatted_text}</div>', unsafe_allow_html=True)
        return
    
    elif method == "fast_words":
        paragraphs = formatted_text.split('\n\n')
        displayed_text = ""
        
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                displayed_text += "\n\n"
            
            words = paragraph.split()
            for i, word in enumerate(words):
                displayed_text += word + " "
                
                if word.endswith(('.', '!', '?')) or i % 3 == 0 or i == len(words) - 1:
                    placeholder.markdown(
                        f'<div class="streaming-text" {base_style}>{displayed_text}<span class="streaming-cursor">▌</span></div>', 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.02)
        
        placeholder.markdown(f'<div class="answer-text" {base_style}>{displayed_text.rstrip()}</div>', unsafe_allow_html=True)
    
    # Add other methods (chunks, sentences, characters) as needed...

def extract_citations_from_answer(answer_text):
    """Extract citation IDs from answer text"""
    if not answer_text:
        return []
    
    citations = set()
    
    # Individual citations
    individual_pattern = r'\[news_([a-f0-9]{6,12})\]'
    individual_matches = re.findall(individual_pattern, answer_text, re.IGNORECASE)
    citations.update(individual_matches)
    
    # Comma-separated citation blocks
    block_pattern = r'\[([news_[a-f0-9]{6,12}(?:,\s*news_[a-f0-9]{6,12})*)\]'
    block_matches = re.findall(block_pattern, answer_text, re.IGNORECASE)
    
    for block in block_matches:
        individual_in_block = re.findall(r'news_([a-f0-9]{6,12})', block, re.IGNORECASE)
        citations.update(individual_in_block)
    
    # Comprehensive fallback
    all_news_pattern = r'news_([a-f0-9]{6,12})'
    all_matches = re.findall(all_news_pattern, answer_text, re.IGNORECASE)
    citations.update(all_matches)
    
    return list(citations)

def display_quality_badge(verdict):
    """Display quality badge based on verdict"""
    classes = {
        "Excellent": "excellent",
        "Good": "good", 
        "Fair": "fair",
        "Poor": "poor"
    }
    badge_class = classes.get(verdict, "fair")
    return f'<div class="quality-badge {badge_class}">{verdict}</div>'

def update_live_stats(stats_placeholder):
    """Update the live statistics display"""
    stats = st.session_state.live_stats
    
    status_class = f"status-{stats['status']}"
    status_text = {
        'ready': '🟢 Ready',
        'running': '🟡 Running',
        'complete': '🟢 Complete',
        'error': '🔴 Error'
    }.get(stats['status'], '🟡 Running')
    
    with stats_placeholder.container():
        st.markdown('<div class="live-stats">', unsafe_allow_html=True)
        st.markdown(f'<div class="status-indicator {status_class}">{status_text}</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⏱️ Time", f"{stats['elapsed_time']:.1f}s")
        with col2:
            st.metric("🔄 Iterations", stats['iterations'])
        with col3:
            st.metric("📑 Articles", stats['articles_found'])
        with col4:
            current_step = stats['current_step']
            if len(current_step) > 20:
                current_step = current_step[:17] + "..."
            st.metric("📍 Step", current_step)
        
        st.markdown('</div>', unsafe_allow_html=True)