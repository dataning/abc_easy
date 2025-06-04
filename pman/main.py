import pandas as pd
import warnings
import os
import importlib
import sys
from datetime import datetime

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.layout import Layout
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not available - progress display will be basic")
    print("💡 Install with: pip install rich")

# Auto-reload project modules with error handling
def reload_modules():
    """Reload all project modules for development"""
    modules = [
        'base_agent', 'memory', 'tools', 'citation_tracker', 
        'answer_evaluation', 'research_agent', 'research_planning',
        'enhanced_keyword_extractor'
    ]
    reloaded = []
    failed = []
    
    for module in modules:
        try:
            if module in sys.modules:
                importlib.reload(sys.modules[module])
                reloaded.append(module)
        except Exception as e:
            failed.append((module, str(e)))
    
    if RICH_AVAILABLE:
        if reloaded:
            console.print(f"🔄 Reloaded modules: {', '.join(reloaded)}", style="dim green")
        if failed:
            for mod, err in failed:
                console.print(f"⚠️ Failed to reload {mod}: {err}", style="dim red")
    else:
        if reloaded:
            print(f"🔄 Reloaded: {', '.join(reloaded)}")
        if failed:
            for mod, err in failed:
                print(f"⚠️ Failed to reload {mod}: {err}")

# Call before imports
reload_modules()

# Import all required modules
from research_agent import ResearchAgent
from memory import EnhancedMemory
from tools import LLMTool, NewsSearchTool, SemanticFilterTool, RelevanceEvaluatorTool
from citation_tracker import CitationTracker
from answer_evaluation import AnswerEvaluator, AnswerRegenerator, EvaluationResult

def print_banner():
    """Print a nice banner for the application"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if RICH_AVAILABLE:
        # Rich banner
        banner_text = f"""
[bold blue]🤖 HIERARCHICAL RESEARCH AGENT[/bold blue] [dim](v1.3)[/dim]
Session started: {current_time} (Local Time)
Mode: Enhanced Memory + Multi-Provider LLM | Auto-Reload: ON
Data Storage: Memory Only (No Files Written)
        """
        console.print(Panel(banner_text.strip(), border_style="blue"))
    else:
        # Simple banner
        try:
            import shutil
            width = min(shutil.get_terminal_size().columns, 80)
        except:
            width = 80
        
        box_width = width - 2
        bottom_line = "└" + "─" * (box_width - 1)
        
        print(f"\n┌─🤖 HIERARCHICAL RESEARCH AGENT (v1.3)")
        print(f"│ Session started: {current_time} (Local Time)")
        print(f"│ Mode: Enhanced Memory + Multi-Provider LLM | Auto-Reload: ON")
        print(f"│ Data Storage: Memory Only (No Files Written)")
        print(bottom_line)

def get_research_question():
    """Get research question from user with validation"""
    while True:
        if RICH_AVAILABLE:
            console.print("\n📝 Enter your research question:", style="bold cyan")
        else:
            print("\n📝 **Enter your research question:**")
        
        question = input(">>> ").strip()
        
        if len(question) < 5:
            if RICH_AVAILABLE:
                console.print("⚠️ Question too short. Please provide more detail.", style="yellow")
            else:
                print("⚠️ Question too short. Please provide more detail.")
            continue
            
        if len(question) > 200:
            if RICH_AVAILABLE:
                console.print("⚠️ Question too long. Please be more concise.", style="yellow")
            else:
                print("⚠️ Question too long. Please be more concise.")
            continue
        
        return question

def print_answer_section(results):
    """Print the main answer section"""
    answer = results.get('answer', 'No answer generated')
    
    if RICH_AVAILABLE:
        console.print("\n" + "="*75, style="bold blue")
        console.print("📊 RESEARCH RESULTS", style="bold blue")
        console.print("="*75, style="bold blue")
        
        console.print("\n📝 [bold]ANSWER:[/bold]")
        console.print("-" * 60, style="dim")
        
        # Format answer with proper line breaks
        formatted_answer = answer.replace('\n\n', '\n').strip()
        console.print(formatted_answer, style="white")
        console.print("-" * 60, style="dim")
    else:
        print("\n" + "=" * 75)
        print("📊 RESEARCH RESULTS")
        print("=" * 75)
        
        print(f"\n📝 **ANSWER:**")
        print("-" * 60)
        print(answer)
        print("-" * 60)

def print_quality_assessment(results):
    """Print answer quality assessment"""
    eval_stats = results.get("evaluation_stats", {})
    
    verdict = eval_stats.get('final_verdict', 'Unknown')
    score = eval_stats.get('final_score', 0.0)
    attempts = eval_stats.get('improvement_attempts', 0)
    
    # Color and icon mapping
    verdict_colors = {
        'Excellent': 'bright_green',
        'Good': 'yellow', 
        'Fair': 'orange3',
        'Poor': 'red'
    }
    verdict_icons = {
        'Excellent': '🟢',
        'Good': '🟡', 
        'Fair': '🟠',
        'Poor': '🔴'
    }
    
    verdict_icon = verdict_icons.get(verdict, '⚪')
    
    if RICH_AVAILABLE:
        console.print(f"\n🎯 [bold]ANSWER QUALITY ASSESSMENT:[/bold]")
        
        verdict_color = verdict_colors.get(verdict, 'white')
        console.print(f"   {verdict_icon} Final verdict: [bold {verdict_color}]{verdict}[/bold {verdict_color}]")
        console.print(f"   📊 Quality score: [bold]{score:.2f}/1.0[/bold] ({score*100:.0f}%)")
        console.print(f"   🔄 Improvement attempts: [bold]{attempts}[/bold]")
        
        # Performance indicator
        if score >= 0.8:
            console.print("   🎉 Excellent research quality achieved!", style="bright_green")
        elif score >= 0.6:
            console.print("   👍 Good research quality", style="yellow")
        elif score >= 0.4:
            console.print("   📈 Acceptable research quality", style="orange3")
        else:
            console.print("   🔧 Research quality needs improvement", style="red")
    else:
        print(f"\n🎯 **ANSWER QUALITY ASSESSMENT:**")
        print(f"   {verdict_icon} Final verdict: **{verdict}**")
        print(f"   📊 Quality score: **{score:.2f}/1.0** ({score*100:.0f}%)")
        print(f"   🔄 Improvement attempts: **{attempts}**")
        
        # Performance indicator
        if score >= 0.8:
            print("   🎉 Excellent research quality achieved!")
        elif score >= 0.6:
            print("   👍 Good research quality")
        elif score >= 0.4:
            print("   📈 Acceptable research quality")
        else:
            print("   🔧 Research quality needs improvement")

def print_citation_statistics(results):
    """Print detailed citation statistics"""
    citation_stats = results.get("citation_stats", {})
    
    total = citation_stats.get('total_tracked', 0)
    used = citation_stats.get('used_in_answer', 0)
    hallucinations = citation_stats.get('hallucinations', 0)
    
    if RICH_AVAILABLE:
        console.print(f"\n🔍 [bold]CITATION ANALYSIS:[/bold]")
        console.print(f"   📚 Total articles retrieved: [bold]{total}[/bold]")
        console.print(f"   ✅ Citations used in answer: [bold]{used}[/bold]")
        console.print(f"   ❌ Hallucinations detected: [bold]{hallucinations}[/bold]")
        
        if total > 0:
            usage_rate = (used / total) * 100
            console.print(f"   📊 Citation usage rate: [bold]{usage_rate:.1f}%[/bold]")
        
        if hallucinations == 0:
            console.print("   🎯 Perfect citation accuracy!", style="bright_green")
        elif hallucinations <= 2:
            console.print("   ⚠️ Minor citation issues detected", style="yellow")
        else:
            console.print("   🚨 Multiple citation issues found", style="red")
    else:
        print(f"\n🔍 **CITATION ANALYSIS:**")
        print(f"   📚 Total articles retrieved: **{total}**")
        print(f"   ✅ Citations used in answer: **{used}**")
        print(f"   ❌ Hallucinations detected: **{hallucinations}**")
        
        if total > 0:
            usage_rate = (used / total) * 100
            print(f"   📊 Citation usage rate: **{usage_rate:.1f}%**")
        
        if hallucinations == 0:
            print("   🎯 Perfect citation accuracy!")
        elif hallucinations <= 2:
            print("   ⚠️ Minor citation issues detected")
        else:
            print("   🚨 Multiple citation issues found")

def print_citations_used(results):
    """Print detailed list of citations used"""
    used_citations = results.get("used_citations", [])
    
    if used_citations:
        if RICH_AVAILABLE:
            console.print(f"\n📖 [bold]CITATIONS USED ({len(used_citations)}):[/bold]")
            
            # Create citation table
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("ID", style="dim", width=12)
            table.add_column("Title", style="white", width=50)
            table.add_column("Domain", style="cyan", width=20)
            table.add_column("Date", style="green", width=12)
            
            for citation in used_citations[:10]:  # Show first 10
                table.add_row(
                    citation.get('news_id', 'N/A'),
                    citation.get('title', 'No title')[:47] + "..." if len(citation.get('title', '')) > 50 else citation.get('title', 'No title'),
                    citation.get('domain', 'Unknown'),
                    citation.get('date', 'Unknown')
                )
            
            console.print(table)
            
            if len(used_citations) > 10:
                console.print(f"   ... and {len(used_citations) - 10} more citations", style="dim")
        else:
            print(f"\n📖 **CITATIONS USED ({len(used_citations)}):**")
            for i, citation in enumerate(used_citations[:10], 1):
                title = citation.get('title', 'No title')
                if len(title) > 60:
                    title = title[:57] + "..."
                print(f"   {i:2d}. {title}")
                print(f"       🌐 {citation.get('domain', 'Unknown')} | 📅 {citation.get('date', 'Unknown')}")
            
            if len(used_citations) > 10:
                print(f"   ... and {len(used_citations) - 10} more citations")
    else:
        if RICH_AVAILABLE:
            console.print(f"\n📖 [bold]CITATIONS USED:[/bold] [dim]None[/dim]")
        else:
            print(f"\n📖 **CITATIONS USED:** None")

def print_hierarchical_progress(results):
    """Print detailed hierarchical subgoal progress"""
    progress_report = results.get("progress_report", {})
    
    if RICH_AVAILABLE:
        console.print(f"\n🎯 [bold]HIERARCHICAL PROGRESS:[/bold]")
        
        # Create progress table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Phase", style="cyan", width=25)
        table.add_column("Status", width=15)
        table.add_column("Progress", width=30)
        table.add_column("Details", style="dim", width=25)
        
        phases = progress_report.get("subgoal_completion", {})
        for phase, data in phases.items():
            status = "✅ Complete" if data.get("completed", False) else "🔄 In Progress"
            status_style = "green" if data.get("completed", False) else "yellow"
            
            completed_steps = data.get("completed_steps", 0)
            total_steps = data.get("total_steps", 1)
            progress_bar = "█" * int(completed_steps/total_steps * 10) + "▒" * (10 - int(completed_steps/total_steps * 10))
            progress_text = f"{progress_bar} {completed_steps}/{total_steps}"
            
            details = f"{data.get('success_count', 0)} successes"
            
            table.add_row(
                phase.replace("_", " ").title(),
                f"[{status_style}]{status}[/{status_style}]",
                progress_text,
                details
            )
        
        console.print(table)
        
        # Overall stats
        total_iterations = progress_report.get("total_iterations", 0)
        planning_failures = progress_report.get("planning_failures", 0)
        console.print(f"\n   📊 Total iterations: [bold]{total_iterations}[/bold] | Planning failures: [bold]{planning_failures}[/bold]")
    else:
        print(f"\n🎯 **HIERARCHICAL PROGRESS:**")
        phases = progress_report.get("subgoal_completion", {})
        for phase, data in phases.items():
            status = "✅ Complete" if data.get("completed", False) else "🔄 In Progress"
            completed_steps = data.get("completed_steps", 0)
            total_steps = data.get("total_steps", 1)
            print(f"   {phase.replace('_', ' ').title()}: {status} ({completed_steps}/{total_steps} steps)")

def print_agent_performance(results):
    """Print agent performance metrics"""
    memory_summary = results.get("memory_summary", {})
    
    if RICH_AVAILABLE:
        console.print(f"\n⚡ [bold]AGENT PERFORMANCE:[/bold]")
        
        metrics = [
            ("Final confidence", f"{results.get('final_confidence', 0):.1%}"),
            ("Memory items", str(memory_summary.get('total_items', 0))),
            ("Learning patterns", str(memory_summary.get('patterns_learned', 0))),
            ("Search strategies", str(memory_summary.get('strategies_tested', 0)))
        ]
        
        for metric, value in metrics:
            console.print(f"   📈 {metric}: [bold]{value}[/bold]")
    else:
        print(f"\n⚡ **AGENT PERFORMANCE:**")
        print(f"   📈 Final confidence: **{results.get('final_confidence', 0):.1%}**")
        print(f"   📈 Memory items: **{memory_summary.get('total_items', 0)}**")
        print(f"   📈 Learning patterns: **{memory_summary.get('patterns_learned', 0)}**")
        print(f"   📈 Search strategies: **{memory_summary.get('strategies_tested', 0)}**")

def print_memory_insights(results):
    """Print memory insights and learning patterns"""
    memory_summary = results.get("memory_summary", {})
    
    insights = memory_summary.get("key_insights", [])
    if insights:
        if RICH_AVAILABLE:
            console.print(f"\n🧠 [bold]MEMORY INSIGHTS:[/bold]")
            for insight in insights[:5]:
                console.print(f"   💡 {insight}", style="dim")
        else:
            print(f"\n🧠 **MEMORY INSIGHTS:**")
            for insight in insights[:5]:
                print(f"   💡 {insight}")

def print_session_summary(articles_df, results):
    """Print final session summary"""
    total_articles = len(articles_df) if not articles_df.empty else 0
    citation_stats = results.get("citation_stats", {})
    eval_stats = results.get("evaluation_stats", {})
    
    if RICH_AVAILABLE:
        console.print(f"\n" + "="*75, style="bold green")
        console.print("📋 SESSION SUMMARY", style="bold green")
        console.print("="*75, style="bold green")
        
        # Research stats
        console.print(f"\n📊 [bold]Research Statistics:[/bold]")
        console.print(f"   • Articles processed: [bold]{total_articles}[/bold]")
        console.print(f"   • Citations used: [bold]{citation_stats.get('used_in_answer', 0)}[/bold]")
        console.print(f"   • Quality score: [bold]{eval_stats.get('final_score', 0):.2f}/1.0[/bold]")
        console.print(f"   • Iterations: [bold]{results.get('iterations', 0)}[/bold]")
        
        # Data storage info
        console.print(f"\n💾 [bold]Data Storage:[/bold]")
        if total_articles > 0:
            console.print(f"   • Research data available in memory", style="green")
            console.print(f"   • Citation tracking active", style="green")
            console.print(f"   • No files written to disk", style="cyan")
        
        console.print(f"\n⏰ Session completed: [bold]{datetime.now().strftime('%H:%M:%S')}[/bold]")
        console.print("="*75, style="bold green")
    else:
        print(f"\n" + "=" * 75)
        print("📋 SESSION SUMMARY")
        print("=" * 75)
        
        # Research stats
        print(f"\n📊 **Research Statistics:**")
        print(f"   • Articles processed: **{total_articles}**")
        print(f"   • Citations used: **{citation_stats.get('used_in_answer', 0)}**")
        print(f"   • Quality score: **{eval_stats.get('final_score', 0):.2f}/1.0**")
        print(f"   • Iterations: **{results.get('iterations', 0)}**")
        
        # Data storage info
        print(f"\n💾 **Data Storage:**")
        if total_articles > 0:
            print(f"   • Research data available in memory")
            print(f"   • Citation tracking active")
            print(f"   • No files written to disk")
        
        print(f"\n⏰ Session completed: **{datetime.now().strftime('%H:%M:%S')}**")
        print("=" * 75)

def main():
    """Main function with comprehensive workflow"""
    try:
        # Print application banner
        print_banner()
        
        if RICH_AVAILABLE:
            console.print("🔧 Initializing research tools...", style="bold blue")
        else:
            print("\n🔧 Initializing research tools...")
            
        # Initialize tools with progress indication
        tools = [
            LLMTool(),
            NewsSearchTool(), 
            SemanticFilterTool(),
            RelevanceEvaluatorTool()
        ]
        
        if RICH_AVAILABLE:
            console.print("✅ Tools initialized successfully", style="green")
        else:
            print("✅ Tools initialized successfully")
        
        # Create agent with enhanced memory
        if RICH_AVAILABLE:
            console.print("🧠 Setting up enhanced memory system...", style="bold blue")
        else:
            print("🧠 Setting up enhanced memory system...")
            
        memory = EnhancedMemory()
        agent = ResearchAgent("HierarchicalResearcher", tools, memory)
        
        if RICH_AVAILABLE:
            console.print("✅ Agent ready for research", style="bold green")
        else:
            print("✅ Agent ready for research")
        
        # Get research question from user
        question = get_research_question()
        
        # Execute research with progress indication
        if RICH_AVAILABLE:
            console.print(f"\n🚀 Starting hierarchical research...", style="bold cyan")
            console.print(f"📝 Research question: [italic]{question}[/italic]")
        else:
            print(f"\n🚀 **Starting hierarchical research...**")
            print(f"📝 Research question: *{question}*")
        
        if not RICH_AVAILABLE:
            print("\n" + "-" * 60)
        
        # Main research execution
        articles_df, results = agent.execute_research_with_chaining(question)
        
        # Clear progress line and add spacing
        if not RICH_AVAILABLE:
            print("\n" + "=" * 60)

        # Print memory contents for debugging
        if RICH_AVAILABLE:
            console.print("\n🔍 Memory Contents (Debug):", style="bold yellow")
        else:
            print("\n🔍 **Memory Contents (Debug):**")
        agent.memory.print_memory_snapshot()
        
        # Print all result sections
        print_answer_section(results)
        print_quality_assessment(results)
        print_citation_statistics(results)
        print_citations_used(results)
        print_hierarchical_progress(results)
        print_agent_performance(results)
        print_memory_insights(results)
        print_session_summary(articles_df, results)
        
        return articles_df, results
        
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n\n⏹️ Research interrupted by user", style="yellow")
        else:
            print("\n\n⏹️ Research interrupted by user")
        return pd.DataFrame(), {}
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n❌ [bold red]Error during research:[/bold red] {str(e)}")
            console.print("🔧 Try reloading modules or check your configuration", style="dim")
        else:
            print(f"\n❌ **Error during research:** {str(e)}")
            print("🔧 Try reloading modules or check your configuration")
        return pd.DataFrame(), {}

if __name__ == "__main__":
    # Run the main research workflow
    research_data, analysis_results = main()
    
    # Keep results in memory only (no file saving)
    if not research_data.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if RICH_AVAILABLE:
            console.print(f"\n💾 Research session data available in variables:", style="bold green")
            console.print(f"   • [cyan]research_data[/cyan]: DataFrame with {len(research_data)} articles")
            console.print(f"   • [cyan]analysis_results[/cyan]: Dict with complete analysis")
            console.print(f"   • [dim]No files written to disk[/dim]")
        else:
            print(f"\n💾 Research session data available in variables:")
            print(f"   • research_data: DataFrame with {len(research_data)} articles")
            print(f"   • analysis_results: Dict with complete analysis")
            print(f"   • No files written to disk")