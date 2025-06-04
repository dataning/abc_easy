# agent.py - Core agent architecture and base classes

import re
import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from memory import Memory, EnhancedMemory, Thought, Action, Observation
from citation_tracker import CitationTracker  # Fixed import
from answer_evaluation import EvaluationResult, AnswerEvaluator, AnswerRegenerator  # Fixed import
from tools import Tool, LLMTool, NewsSearchTool, SemanticFilterTool, RelevanceEvaluatorTool

class Agent(ABC):
    """Base agent class with planning, action, reflection cycle"""
    def __init__(self, name: str, tools: List[Tool], memory: Memory = None):
        self.name = name
        self.tools = {t.name(): t for t in tools}
        self.memory = memory or Memory()
        self.thoughts: List[Thought] = []
        self.actions: List[Action] = []
        self.observations: List[Observation] = []

    @abstractmethod
    def think(self, context: Dict) -> Thought:
        pass

    @abstractmethod
    def plan(self, goal: str, context: Dict) -> List[Action]:
        pass

    def act(self, action: Action) -> Observation:
        if action.name not in self.tools:
            return Observation(action, None, False, [f"Unknown tool: {action.name}"])
        try:
            result = self.tools[action.name].execute(**action.params)
            obs = Observation(action, result, True)
            self.observations.append(obs)
            return obs
        except Exception as e:
            return Observation(action, str(e), False, [f"Error: {e}"])

    def reflect(self) -> List[str]:
        learnings = []
        recent = self.observations[-10:]
        success_rate = sum(1 for o in recent if o.success) / len(recent) if recent else 0
        if success_rate < 0.5:
            learnings.append(f"Low success rate ({success_rate:.1%}). Need to adjust strategy.")
        
        failures = [o for o in recent if not o.success]
        if failures:
            errs = {}
            for f in failures:
                et = f.result.split(":")[0] if isinstance(f.result, str) else "Unknown"
                errs[et] = errs.get(et, 0) + 1
            mc = max(errs.items(), key=lambda x: x[1])
            learnings.append(f"Most common failure: {mc[0]} ({mc[1]} times)")
        
        for l in learnings:
            self.memory.reflect("performance_analysis", l)
        return learnings

    def print_memory_contents(self):
        """Print detailed contents of agent's memory for debugging"""
        print("\n" + "=" * 60)
        print("🧠 AGENT MEMORY CONTENTS")
        print("=" * 60)
        
        # Short-term memory
        print("\n📋 Short-term Memory:")
        if not self.memory.short_term:
            print("  (empty)")
        else:
            for i, item in enumerate(self.memory.short_term[-10:], 1):
                key = item["key"]
                value = item["value"]
                importance = item["importance"]
                timestamp = item["timestamp"].strftime("%H:%M:%S")
                
                if isinstance(value, pd.DataFrame):
                    value_str = f"DataFrame({len(value)} rows, {len(value.columns)} cols)"
                elif isinstance(value, list):
                    value_str = f"List({len(value)} items): {str(value)[:50]}..."
                elif isinstance(value, dict):
                    value_str = f"Dict({len(value)} keys): {list(value.keys())}"
                elif isinstance(value, str):
                    value_str = f"'{value[:50]}{'...' if len(value) > 50 else ''}'"
                else:
                    value_str = str(value)[:50]
                
                print(f"  {i:2d}. [{timestamp}] {key} (importance: {importance:.1f})")
                print(f"      {value_str}")
        
        # Long-term memory
        print("\n🏛️ Long-term Memory:")
        if not self.memory.long_term:
            print("  (empty)")
        else:
            for key, value in self.memory.long_term.items():
                if isinstance(value, pd.DataFrame):
                    value_str = f"DataFrame({len(value)} rows, {len(value.columns)} cols)"
                    if not value.empty and "title" in value.columns:
                        sample_titles = value["title"].head(3).tolist()
                        value_str += f"\n      Sample titles: {sample_titles}"
                elif isinstance(value, list):
                    value_str = f"List({len(value)} items): {value}"
                elif isinstance(value, dict):
                    value_str = f"Dict: {value}"
                else:
                    value_str = str(value)[:100]
                
                print(f"  • {key}:")
                print(f"    {value_str}")
        
        # Reflections
        print("\n💭 Reflections:")
        if not self.memory.reflections:
            print("  (empty)")
        else:
            for i, reflection in enumerate(self.memory.reflections[-5:], 1):
                timestamp = reflection["timestamp"].strftime("%H:%M:%S")
                print(f"  {i}. [{timestamp}] {reflection['experience']}")
                print(f"     Lesson: {reflection['lesson']}")
        
        # Agent-specific extensions
        if hasattr(self, '_print_agent_specific_memory'):
            self._print_agent_specific_memory()

    def save_memory_to_file(self, filename: str = "agent_memory.json"):
        """Save memory contents to a JSON file for analysis"""
        memory_data = {
            "timestamp": datetime.now().isoformat(),
            "short_term": [],
            "long_term": {},
            "reflections": [],
            "agent_state": {},
        }
        
        # Convert memory (handle non-serializable objects)
        for item in self.memory.short_term:
            value = item["value"]
            if isinstance(value, pd.DataFrame):
                value = f"DataFrame({len(value)} rows, {len(value.columns)} columns)"
            elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                value = str(value)
            
            memory_data["short_term"].append({
                "key": item["key"],
                "value": value,
                "importance": item["importance"],
                "timestamp": item["timestamp"].isoformat()
            })
        
        for key, value in self.memory.long_term.items():
            if isinstance(value, pd.DataFrame):
                memory_data["long_term"][key] = f"DataFrame({len(value)} rows, {len(value.columns)} columns)"
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                memory_data["long_term"][key] = value
            else:
                memory_data["long_term"][key] = str(value)
        
        for reflection in self.memory.reflections:
            memory_data["reflections"].append({
                "experience": reflection["experience"],
                "lesson": reflection["lesson"],
                "timestamp": reflection["timestamp"].isoformat()
            })
        
        with open(filename, 'w') as f:
            json.dump(memory_data, f, indent=2)
        
        print(f"💾 Memory saved to {filename}")


# Factory function for backward compatibility
def create_research_agent(name: str, tools, memory=None):
    """Factory function to create a research agent with enhanced memory"""
    from research_agent import ResearchAgent  # Import here to avoid circular imports
    return ResearchAgent(name, tools, memory or EnhancedMemory())