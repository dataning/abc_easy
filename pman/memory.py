import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Rich library support with graceful fallback
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

@dataclass
class Thought:
    content: str
    confidence: float
    timestamp: datetime = datetime.now()

@dataclass
class Action:
    name: str
    params: Dict[str, Any]
    expected_outcome: str

@dataclass
class Observation:
    action: Action
    result: Any
    success: bool
    learnings: List[str] = None

class Memory:
    """Unified memory system for agents"""
    def __init__(self):
        self.short_term: List[Dict] = []
        self.long_term: Dict[str, Any] = {}
        self.reflections: List[Dict] = []

    def remember(self, key: str, value: Any, importance: float = 0.5):
        item = {
            "key": key,
            "value": value,
            "importance": importance,
            "timestamp": datetime.now()
        }
        self.short_term.append(item)
        if importance > 0.7:
            self.long_term[key] = value

    def recall(self, key: str) -> Optional[Any]:
        for item in reversed(self.short_term):
            if item["key"] == key:
                return item["value"]
        return self.long_term.get(key)

    def reflect(self, experience: str, lesson: str):
        self.reflections.append({
            "experience": experience,
            "lesson": lesson,
            "timestamp": datetime.now()
        })

    def _format_size(self, value: Any) -> str:
        """Format memory size in a human-readable way"""
        if isinstance(value, pd.DataFrame):
            rows, cols = value.shape
            return f"{rows}×{cols}"
        elif isinstance(value, (list, tuple)):
            return f"{len(value)} items"
        elif isinstance(value, dict):
            size_bytes = len(str(value).encode('utf-8'))
            if size_bytes < 1024:
                return f"{size_bytes} B"
            else:
                return f"{size_bytes//1024} KB"
        elif isinstance(value, str):
            size_bytes = len(value.encode('utf-8'))
            if size_bytes < 1024:
                return f"{size_bytes} B"
            else:
                return f"{size_bytes//1024} KB"
        else:
            return "N/A"

    def _format_summary(self, key: str, value: Any) -> str:
        """Create a concise summary of the memory value"""
        if isinstance(value, pd.DataFrame):
            if not value.empty and 'title' in value.columns:
                first_title = value.iloc[0]['title'][:25] + "..." if len(value.iloc[0]['title']) > 25 else value.iloc[0]['title']
                return f'Titles include "{first_title}"'
            else:
                return f"{len(value)} articles"
        elif isinstance(value, dict):
            if 'confidence' in value or 'score' in value:
                parts = []
                if 'confidence' in value:
                    parts.append(f"confidence:{value['confidence']:.2f}")
                if 'score' in value:
                    parts.append(f"score:{value['score']:.2f}")
                if 'relevant' in value:
                    parts.append(f"relevant:{value['relevant']}")
                return "; ".join(parts) + "…"
            else:
                keys = list(value.keys())[:3]
                return f"Keys: {', '.join(keys)}…"
        elif isinstance(value, list):
            if value and isinstance(value[0], str):
                preview = ', '.join(value[:3])
                return f'[{preview}{"…" if len(value) > 3 else ""}]'
            else:
                return f"{len(value)} items"
        elif isinstance(value, str):
            return f'"{value[:40]}{"…" if len(value) > 40 else ""}"'
        else:
            return str(value)[:40] + ("…" if len(str(value)) > 40 else "")

    def _get_value_type(self, value: Any) -> str:
        """Get a human-readable type name"""
        if isinstance(value, pd.DataFrame):
            return "DataFrame"
        elif isinstance(value, dict):
            return "dict"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, bool):
            return "boolean"
        else:
            return type(value).__name__

    def print_memory_snapshot(self):
        """Print a formatted snapshot of current memory contents"""
        # Collect all memory items
        all_items = {}
        
        # Add short-term memory (latest values for each key)
        seen_keys = set()
        for item in reversed(self.short_term):
            key = item["key"]
            if key not in seen_keys:
                all_items[key] = item["value"]
                seen_keys.add(key)
        
        # Add long-term memory (only if not already in short-term)
        for key, value in self.long_term.items():
            if key not in seen_keys:
                all_items[key] = value

        if RICH_AVAILABLE:
            console = Console()
            table = Table(title="🧠 MEMORY SNAPSHOT (debug only)", border_style="cyan")
            table.add_column("Key", style="bold yellow", width=20)
            table.add_column("Type", style="cyan", width=12)
            table.add_column("Size", style="green", justify="right", width=8)
            table.add_column("Summary", style="dim", width=35)

            for key, value in all_items.items():
                value_type = self._get_value_type(value)
                size = self._format_size(value)
                summary = self._format_summary(key, value)
                
                table.add_row(key, value_type, size, summary)
            
            console.print(table)
            
            # Add reflection summary if available
            if self.reflections:
                console.print(f"\n💭 Reflections: {len(self.reflections)} learning experiences recorded", style="dim cyan")
        else:
            # ASCII table fallback
            print("\n── MEMORY SNAPSHOT (debug only) ──")
            print("│ Key                │ Type          │ Size   │ Summary")
            print("├────────────────────┼───────────────┼────────┼────────────────────────┤")
            
            for key, value in all_items.items():
                value_type = self._get_value_type(value)
                size = self._format_size(value)
                summary = self._format_summary(key, value)
                
                # Truncate fields to fit table width
                key_display = key[:18] + ".." if len(key) > 18 else key.ljust(18)
                type_display = value_type[:13].ljust(13)
                size_display = size[:6].rjust(6)
                summary_display = summary[:22] + ".." if len(summary) > 22 else summary
                
                print(f"│ {key_display} │ {type_display} │ {size_display} │ {summary_display}")
            
            print("└────────────────────┴───────────────┴────────┴────────────────────────┘")
            
            if self.reflections:
                print(f"💭 Reflections: {len(self.reflections)} learning experiences recorded")

class EnhancedMemory(Memory):
    """Enhanced memory with strategic learning capabilities"""
    def __init__(self):
        super().__init__()
        self.search_strategy = {}
        self.domain_knowledge = {}
        self.error_patterns = {}
        self.performance_metrics = {}
    
    def track_search_effectiveness(self, keywords: List[str], relevance: float, article_count: int):
        """Track which search strategies work"""
        strategy_key = "+".join(sorted(keywords))
        if strategy_key not in self.search_strategy:
            self.search_strategy[strategy_key] = []
        
        self.search_strategy[strategy_key].append({
            "relevance": relevance,
            "articles": article_count,
            "timestamp": datetime.now()
        })
    
    def learn_domain_patterns(self, question: str, successful_keywords: List[str], reliable_sources: List[str]):
        """Build domain-specific knowledge"""
        domain = self._extract_domain(question)
        if domain not in self.domain_knowledge:
            self.domain_knowledge[domain] = {
                "keywords": set(),
                "sources": set(),
                "patterns": []
            }
        
        self.domain_knowledge[domain]["keywords"].update(successful_keywords)
        self.domain_knowledge[domain]["sources"].update(reliable_sources)
    
    def prevent_repeated_errors(self, error_type: str, context: Dict):
        """Learn from failures to prevent repetition"""
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = []
        
        self.error_patterns[error_type].append({
            "context": context,
            "timestamp": datetime.now(),
            "prevention_strategy": self._suggest_prevention(error_type)
        })
    
    def _extract_domain(self, question: str) -> str:
        """Extract domain from question for pattern learning"""
        question_lower = question.lower()
        if "harvard" in question_lower or "university" in question_lower:
            return "education"
        elif "economy" in question_lower or "financial" in question_lower:
            return "finance"
        elif "politics" in question_lower or "government" in question_lower:
            return "politics"
        else:
            return "general"
    
    def _suggest_prevention(self, error_type: str) -> str:
        """Suggest prevention strategies for common errors"""
        strategies = {
            "infinite_loop": "Add attempt counters and force progression",
            "low_relevance": "Refine keywords or use semantic filtering",
            "no_articles": "Broaden search terms or try alternative sources",
            "citation_hallucination": "Verify all citations against source material"
        }
        return strategies.get(error_type, "Monitor and adjust strategy")

    def print_memory_snapshot(self):
        """Enhanced memory snapshot including strategic learning data"""
        # Call parent method first
        super().print_memory_snapshot()
        
        # Add enhanced memory specific information
        if RICH_AVAILABLE:
            console = Console()
            
            # Strategic learning summary
            if self.search_strategy or self.domain_knowledge or self.error_patterns:
                learning_table = Table(title="📚 Strategic Learning Data", border_style="yellow")
                learning_table.add_column("Category", style="bold")
                learning_table.add_column("Count", justify="right", style="cyan")
                learning_table.add_column("Details", style="dim")
                
                if self.search_strategy:
                    learning_table.add_row("Search Strategies", str(len(self.search_strategy)), 
                                         f"Tracking {sum(len(v) for v in self.search_strategy.values())} attempts")
                
                if self.domain_knowledge:
                    domains = list(self.domain_knowledge.keys())
                    learning_table.add_row("Domain Knowledge", str(len(domains)), 
                                         f"Domains: {', '.join(domains)}")
                
                if self.error_patterns:
                    error_count = sum(len(v) for v in self.error_patterns.values())
                    learning_table.add_row("Error Patterns", str(error_count), 
                                         f"Types: {', '.join(self.error_patterns.keys())}")
                
                console.print(learning_table)
        else:
            # ASCII fallback for enhanced data
            if self.search_strategy or self.domain_knowledge or self.error_patterns:
                print("\n📚 Strategic Learning Data:")
                if self.search_strategy:
                    total_attempts = sum(len(v) for v in self.search_strategy.values())
                    print(f"   🔍 Search strategies: {len(self.search_strategy)} (tracking {total_attempts} attempts)")
                
                if self.domain_knowledge:
                    domains = list(self.domain_knowledge.keys())
                    print(f"   🎓 Domain knowledge: {len(domains)} domains ({', '.join(domains)})")
                
                if self.error_patterns:
                    error_count = sum(len(v) for v in self.error_patterns.values())
                    print(f"   ⚠️ Error patterns: {error_count} tracked ({', '.join(self.error_patterns.keys())})")