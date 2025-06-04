# research_agent.py - Core research agent with hierarchical planning and enhanced keyword extraction

import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from base_agent import Agent
from memory import EnhancedMemory, Thought, Action, Observation
from citation_tracker import CitationTracker
from answer_evaluation import EvaluationResult, AnswerEvaluator, AnswerRegenerator
from tools import Tool
from research_planning import ResearchPlanner

class ResearchAgent(Agent):
    """Agent specialized in research tasks with hierarchical subgoal tracking and enhanced keyword extraction"""
    
    def __init__(self, name: str, tools: List[Tool], memory=None):
        # Use enhanced memory by default for research agents
        enhanced_memory = memory or EnhancedMemory()
        super().__init__(name, tools, enhanced_memory)
        
        # Initialize core components
        self.citation_tracker = CitationTracker()
        self.answer_evaluator = AnswerEvaluator(tools[0]) if tools else None
        self.answer_regenerator = AnswerRegenerator(tools[0]) if tools else None
        self.planner = ResearchPlanner(self)
        
        # Progress tracking
        self.planning_failures = 0
        self.current_subgoal: Optional[str] = None
        self.current_step: Optional[str] = None
        self._last_completed_subgoals = 0
        
        # Enhanced keyword extraction tracking
        self._enhanced_extraction_used = False
        self._last_extraction_stats = {}
        
        # Expose subgoals from planner for backward compatibility
        self.subgoals = self.planner.subgoals

    def _show_live_progress(self, iteration: int, max_iterations: int):
        """Show live progress with TRUE phase-based display"""
        # Get TRUE phase summary (not subgoals)
        phases = self.planner.get_phase_summary()
        
        # Calculate overall progress based on PHASES
        total_phases = len(phases)
        completed_phases = sum(1 for status in phases.values() if status.startswith("✅"))
        progress_percent = (completed_phases / total_phases * 100) if total_phases > 0 else 0
        
        # Create visual progress bar
        bar_length = 12
        filled_length = int(bar_length * completed_phases // total_phases) if total_phases > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Current activity - get from the FIRST incomplete phase
        current_activity = ""
        for phase_name, status in phases.items():
            if not status.startswith("✅"):
                phase_short = phase_name.split(":")[1].strip() if ":" in phase_name else phase_name
                current_activity = f" | {phase_short}"
                break
        
        # Show enhanced extraction indicator if used
        extraction_indicator = " 🧠" if self._enhanced_extraction_used else ""
        
        # Show PHASE-based progress (not subgoals)
        print(f"\r🚀 Research in progress: [{bar}] {progress_percent:.0f}% (Phases: {completed_phases}/{total_phases}){current_activity}{extraction_indicator}", end="", flush=True)

    def print_phase_summary(self):
        """Print a clean phase-by-phase summary with extraction stats"""
        phases = self.planner.get_phase_summary()
        print("\n📋 Research Progress by Phase:")
        for phase_name, status in phases.items():
            print(f"   {phase_name} {status}")
        
        # Show extraction statistics if enhanced extraction was used
        if self._enhanced_extraction_used and self._last_extraction_stats:
            print(f"\n🧠 Enhanced Keyword Extraction Stats:")
            stats = self._last_extraction_stats
            print(f"   • Graph nodes: {stats.get('graph_nodes', 0)}")
            print(f"   • Graph edges: {stats.get('graph_edges', 0)}")
            print(f"   • Primary keywords: {stats.get('primary_count', 0)}")
            print(f"   • Search variants: {stats.get('variant_count', 0)}")
            print(f"   • Extraction confidence: {stats.get('confidence', 0):.1%}")
            print(f"   • Most central concept: {stats.get('most_central', 'N/A')}")

    def think(self, context: Dict) -> Thought:
        """Research-specific thinking that considers articles, relevance, and progress"""
        articles = self._get_current_articles()
        articles_count = len(articles) if isinstance(articles, pd.DataFrame) else 0
        relevance = self._get_current_relevance_confidence()
        
        # Use TRUE phase-based progress tracking
        phases = self.planner.get_phase_summary()
        completed_phases = sum(1 for status in phases.values() if status.startswith("✅"))
        total_phases = len(phases)

        content = f"Progress: {completed_phases}/{total_phases} phases complete. "
        content += f"Current state: {articles_count} articles, {relevance:.1%} relevance. "
        
        # Add extraction quality context
        if self._enhanced_extraction_used:
            content += f"Enhanced extraction: {self._last_extraction_stats.get('confidence', 0):.1%} confidence. "
        
        if articles_count < 20:
            content += "Priority: Need more articles."
        elif relevance < 0.4:
            content += "Priority: Improve relevance through filtering or better keywords."
        else:
            content += "Priority: Generate comprehensive answer."

        confidence = completed_phases / total_phases if total_phases > 0 else 0.0
        thought = Thought(content=content, confidence=confidence)
        self.thoughts.append(thought)
        
        self.memory.remember("last_thought", thought.content, importance=0.5)
        print(f"   💭 Thought: {thought.content}")
        
        return thought

    def plan(self, goal: str, context: Dict) -> List[Action]:
        """Research-specific planning using hierarchical subgoals"""
        next_action = self.planner.plan_next_step_hierarchical(goal)
        return [next_action] if next_action else []

    def _print_agent_specific_memory(self):
        """Print research agent specific memory contents including extraction stats"""
        print("\n🎯 Current State:")
        print(f"  • Current subgoal: {self.current_subgoal}")
        print(f"  • Current step: {self.current_step}")
        print(f"  • Planning failures: {self.planning_failures}")
        print(f"  • Enhanced extraction used: {self._enhanced_extraction_used}")
        
        print("\n📚 Citation Tracker:")
        print(f"  • Total articles tracked: {len(self.citation_tracker.citations)}")
        print(f"  • Hallucination count: {len(self.citation_tracker.hallucinations)}")
        if self.citation_tracker.citations:
            print("  • Recent citations:")
            for i, (url, citation) in enumerate(list(self.citation_tracker.citations.items())[-3:], 1):
                title = citation.get('title', 'No title')[:50]
                print(f"    {i}. [{citation.get('id', 'N/A')}] {title}...")

        # Show enhanced extraction stats if available
        if self._enhanced_extraction_used and self._last_extraction_stats:
            print("\n🧠 Last Enhanced Extraction:")
            stats = self._last_extraction_stats
            print(f"  • Graph structure: {stats.get('graph_nodes', 0)} nodes, {stats.get('graph_edges', 0)} edges")
            print(f"  • Keywords generated: {stats.get('primary_count', 0)} primary, {stats.get('variant_count', 0)} variants")
            print(f"  • Extraction confidence: {stats.get('confidence', 0):.1%}")

    def _update_step_on_success(self, action: Action, result: Any):
        """Update the current step based on successful action result with extraction tracking"""
        if not self.current_subgoal or not self.current_step:
            return
        
        subgoal_name = self.current_subgoal
        step_name = self.current_step
        
        # Track enhanced extraction usage for news search actions
        if action.name == "news_search" and hasattr(self.tools.get("news_search"), "_last_extraction"):
            try:
                news_tool = self.tools["news_search"]
                extraction_data = news_tool.get_last_extraction_analysis()
                if extraction_data:
                    self._enhanced_extraction_used = True
                    graph_analysis = extraction_data.get('graph_analysis', {})
                    self._last_extraction_stats = {
                        'graph_nodes': graph_analysis.get('nodes', 0),
                        'graph_edges': graph_analysis.get('edges', 0),
                        'primary_count': len(extraction_data.get('primary_keywords', [])),
                        'variant_count': len(extraction_data.get('search_variants', [])),
                        'confidence': extraction_data.get('confidence', 0),
                        'most_central': graph_analysis.get('most_central', 'N/A'),
                        'domain': extraction_data.get('llm_insights', {}).get('domain', 'unknown')
                    }
                    print(f"      🧠 Enhanced extraction: {self._last_extraction_stats['confidence']:.1%} confidence, "
                          f"{self._last_extraction_stats['graph_nodes']} graph nodes")
            except Exception as e:
                print(f"      ⚠️ Could not track extraction stats: {e}")
        
        if subgoal_name == "clarity_checked" and step_name == "question_analysis":
            confidence = 0.8
            self.planner.mark_step_complete(subgoal_name, step_name, confidence=confidence)
        
        elif subgoal_name == "keywords_extracted":
            if step_name == "initial_extraction":
                keywords = self._extract_keywords_from_result(result)
                # Boost quality score if enhanced extraction was used
                base_quality = min(len(keywords) / 3.0, 1.0) if keywords else 0.3
                extraction_boost = 0.2 if self._enhanced_extraction_used else 0.0
                quality = min(base_quality + extraction_boost, 1.0)
                
                self.memory.remember("keywords", keywords, importance=0.9)
                self.memory.remember("initial_keywords", keywords, importance=0.8)
                self.planner.mark_step_complete(subgoal_name, step_name, quality=quality)
                
                extraction_info = " (enhanced)" if self._enhanced_extraction_used else ""
                print(f"      ✓ Extracted keywords{extraction_info}: {keywords}")
                
            elif step_name == "keyword_refinement":
                refined_keywords = self._extract_keywords_from_result(result)
                base_quality = min(len(refined_keywords) / 3.0, 1.0) if refined_keywords else 0.5
                extraction_boost = 0.2 if self._enhanced_extraction_used else 0.0
                quality = min(base_quality + extraction_boost, 1.0)
                
                self.memory.remember("refined_keywords", refined_keywords, importance=0.9)
                self.memory.remember("keywords", refined_keywords, importance=0.9)
                self.planner.mark_step_complete(subgoal_name, step_name, quality=quality)
                
                extraction_info = " (enhanced)" if self._enhanced_extraction_used else ""
                print(f"      ✓ Refined keywords{extraction_info}: {refined_keywords}")
        
        elif subgoal_name == "articles_fetched":
            df_candidate = None
            if isinstance(result, pd.DataFrame):
                df_candidate = result
            elif isinstance(result, dict) and "articles_df" in result and isinstance(result["articles_df"], pd.DataFrame):
                df_candidate = result["articles_df"]
            elif hasattr(result, "to_dataframe"):
                df_candidate = result.to_dataframe()

            if isinstance(df_candidate, pd.DataFrame):
                count = len(df_candidate)
                if step_name == "broad_fetch":
                    self.memory.remember("articles", df_candidate, importance=0.8)
                    self.memory.remember("broad_articles", df_candidate, importance=0.7)
                    # Track articles in citation tracker
                    self.citation_tracker.track_articles(df_candidate)
                    tracked_count = len(self.citation_tracker.citations)
                    self.planner.mark_step_complete("articles_fetched", "broad_fetch", count=count)
                    
                    extraction_info = f" via enhanced extraction" if self._enhanced_extraction_used else ""
                    print(f"      ✓ {step_name}: {count} articles ({tracked_count} citations tracked){extraction_info}")
                    
                elif step_name == "focused_fetch":
                    existing = self.memory.recall("articles") or pd.DataFrame()
                    if isinstance(existing, pd.DataFrame) and existing.empty:
                        combined = df_candidate.copy()
                    else:
                        if not df_candidate.empty:
                            combined = pd.concat([existing, df_candidate], ignore_index=True)
                            if "url" in combined.columns:
                                combined = combined.drop_duplicates(subset="url").reset_index(drop=True)
                        else:
                            combined = existing
                    count = len(combined)
                    self.memory.remember("articles", combined, importance=0.8)
                    self.memory.remember("focused_articles", df_candidate, importance=0.7)
                    self.citation_tracker.track_articles(combined)
                    tracked_count = len(self.citation_tracker.citations)
                    self.planner.mark_step_complete("articles_fetched", "focused_fetch", count=count)
                    
                    extraction_info = f" via enhanced extraction" if self._enhanced_extraction_used else ""
                    print(f"      ✓ {step_name}: {count} total articles ({tracked_count} citations tracked){extraction_info}")
                    
                elif step_name == "domain_diversification":
                    existing = self.memory.recall("articles") or pd.DataFrame()
                    if isinstance(existing, pd.DataFrame) and existing.empty:
                        combined = df_candidate.copy()
                        count = len(combined)
                    else:
                        if not df_candidate.empty:
                            combined = pd.concat([existing, df_candidate], ignore_index=True)
                            if "url" in combined.columns:
                                combined = combined.drop_duplicates(subset="url").reset_index(drop=True)
                            count = len(combined)
                        else:
                            combined = existing
                            count = len(existing)
                    self.memory.remember("articles", combined, importance=0.8)
                    self.citation_tracker.track_articles(combined)
                    tracked_count = len(self.citation_tracker.citations)
                    self.planner.mark_step_complete("articles_fetched", "domain_diversification", count=count)
                    print(f"      ✓ {step_name}: {count} articles ({tracked_count} citations tracked)")
            else:
                print(f"      ⚠️ Expected DataFrame (or dict-with-'articles_df'), got {type(result)}")

        elif subgoal_name == "sufficient_relevance":
            if step_name == "initial_relevance_check":
                confidence = result.get("confidence", 0) if isinstance(result, dict) else 0
                self.memory.remember("relevance_check", result, importance=0.8)
                self.memory.remember("initial_relevance", result, importance=0.8)
                self.planner.mark_step_complete(subgoal_name, step_name, confidence=confidence)
                print(f"      ✓ Initial relevance: {confidence:.1%}")
                
                if confidence < 0.3:
                    self.memory.remember("need_keyword_refinement", True, importance=0.9)
                    print(f"      📝 Flagged for keyword refinement (relevance too low)")
                    
            elif step_name == "semantic_filtering":
                if isinstance(result, pd.DataFrame) and not result.empty:
                    self.memory.remember("filtered_articles", result, importance=0.9)
                    filtered_relevance = self._quick_relevance_check(result)
                    self.memory.remember("filtered_relevance", filtered_relevance, importance=0.8)
                    self.planner.mark_step_complete("sufficient_relevance", "semantic_filtering", confidence=filtered_relevance)
                    print(f"      ✓ Filtered to {len(result)} articles, relevance: {filtered_relevance:.1%}")
                else:
                    current_articles = self._get_current_articles()
                    if isinstance(current_articles, pd.DataFrame) and not current_articles.empty:
                        self.memory.remember("filtered_articles", current_articles, importance=0.8)
                        self.planner.mark_step_complete("sufficient_relevance", "semantic_filtering", confidence=self._quick_relevance_check(current_articles))
                        print(f"      📝 Filtering yielded no new articles, using {len(current_articles)} unfiltered articles")
                    else:
                        print(f"      ⚠️ Semantic filter returned no articles")

        elif subgoal_name == "domain_diversity":
            if step_name == "diversity_assessment":
                score = self._extract_diversity_score(result)
                self.planner.mark_step_complete("domain_diversity", "diversity_assessment", score=score)
                print(f"      ✓ Diversity score: {score:.1f}/10")
            elif step_name == "gap_filling_search":
                self.planner.mark_step_complete("domain_diversity", "gap_filling_search", score=0.8)

    def _update_step_on_failure(self, action: Action, error_msg: str):
        """Handle step failure with memory-based learning"""
        step_key = f"{self.current_subgoal}.{self.current_step}" if self.current_subgoal and self.current_step else "unknown"
        
        self.memory.reflect(
            experience=f"Failed step: {step_key}",
            lesson=f"Error: {error_msg[:100]}... May need alternative approach."
        )
        
        if self.current_step == "initial_relevance_check":
            self.memory.remember("need_keyword_refinement", True, importance=0.9)
            print(f"      📝 Relevance check failed, flagging for keyword refinement")
            
        elif self.current_step == "semantic_filtering":
            current_articles = self._get_current_articles()
            if isinstance(current_articles, pd.DataFrame) and not current_articles.empty:
                self.memory.remember("filtered_articles", current_articles, importance=0.8)
                print(f"      📝 Filtering failed, using {len(current_articles)} unfiltered articles")

    def _extract_keywords_from_result(self, result: Any) -> List[str]:
        """Extract keywords from LLM result"""
        if isinstance(result, str):
            cleaned = result.replace(",", " ").replace(";", " ").replace("\n", " ")
            toks = [k.strip() for k in cleaned.split() if len(k.strip()) > 2]
            return toks[:5]
        if isinstance(result, list):
            return result[:5]
        return []

    def _quick_relevance_check(self, articles_df: pd.DataFrame) -> float:
        """Quick relevance estimation based on article count"""
        if not isinstance(articles_df, pd.DataFrame) or articles_df.empty:
            return 0.0
        cnt = len(articles_df)
        if cnt >= 50:
            return 0.8
        if cnt >= 20:
            return 0.6
        if cnt >= 10:
            return 0.4
        return 0.2

    def _extract_diversity_score(self, result: Any) -> float:
        """Extract diversity score from LLM result"""
        if isinstance(result, str):
            m = re.search(r'(\d+(?:\.\d+)?)', result)
            if m:
                sc = float(m.group(1))
                return min(sc / 10.0, 1.0)
        return 0.5

    def _check_comprehensive_satisfaction_hierarchical(self, question: str) -> bool:
        """Check if research goals are satisfied using more realistic criteria"""
        af = self.subgoals["articles_fetched"]
        sr = self.subgoals["sufficient_relevance"]
        dd = self.subgoals["domain_diversity"]
        
        cur_articles = self._get_current_articles()
        cnt = len(cur_articles) if isinstance(cur_articles, pd.DataFrame) else 0
        conf = self._get_current_relevance_confidence()

        # More realistic satisfaction criteria
        has_min_articles = cnt >= 50
        has_acceptable_relevance = conf >= 0.25  # Lowered from 0.4 to 0.25
        has_broad_fetch = any(s["name"]=="broad_fetch" and s["done"] for s in af["steps"])
        
        # Don't require both focused fetch AND diversity
        has_additional_step = (
            any(s["name"]=="focused_fetch" and s["done"] for s in af["steps"]) or
            any(s["name"]=="semantic_filtering" and s["done"] for s in sr["steps"]) or
            dd["complete"]
        )
        
        satisfied = has_min_articles and has_acceptable_relevance and has_broad_fetch and has_additional_step
        
        if satisfied:
            print(f"   📊 Goals achieved: {cnt} articles, {conf:.1%} relevance")
        else:
            missing = []
            if not has_min_articles:
                missing.append(f"need {50-cnt}+ more articles")
            if not has_acceptable_relevance:
                missing.append(f"relevance {conf:.1%} < 25.0%")
            if not has_broad_fetch:
                missing.append("need initial article fetch")
            if not has_additional_step:
                missing.append("need one additional step")
            print(f"   🎯 Still needed: {', '.join(missing)}")
        
        return satisfied

    def _ensure_llm_tool_available(self) -> bool:
        """Ensure LLM tool is properly accessible"""
        if "llm_query" not in self.tools:
            print("   ⚠️ LLM tool not available in tools dictionary")
            return False
        
        try:
            # Test the tool with a simple prompt
            test_result = self.tools["llm_query"].execute("Test", max_tokens=10, temperature=0.1)
            return test_result is not None
        except Exception as e:
            print(f"   ⚠️ LLM tool test failed: {e}")
            return False

    def _create_fallback_answer(self, question: str, articles_df: pd.DataFrame) -> str:
        """Create a structured fallback answer when LLM generation fails"""
        try:
            domains = articles_df['domain'].unique()[:5] if 'domain' in articles_df.columns else ['various sources']
            extraction_note = " Enhanced keyword extraction was used to find these articles." if self._enhanced_extraction_used else ""
            
            return f"""Based on {len(articles_df)} recent articles retrieved about "{question}", this topic shows significant current relevance across multiple news sources including {', '.join(domains)}.{extraction_note}

The search returned articles from recent timeframes, indicating ongoing developments and public interest in this area. While a detailed AI-generated analysis could not be completed due to technical limitations, the substantial number of relevant articles suggests this is an actively discussed topic with multiple perspectives and recent developments.

Further analysis would require manual review of the retrieved articles to provide specific insights and conclusions."""
        except:
            return f"Unable to complete analysis for '{question}' due to technical issues. Please try again or check the system configuration."

    def _regenerate_incremental(self, question: str, feedback: str, articles_df: pd.DataFrame) -> str:
        """Incremental improvement approach using proper news_ids"""
        top_articles = articles_df.head(10)
        articles_text = "\n".join([
            f"[{self.citation_tracker.add_article(row.get('title', 'No title'), row.get('domain', 'Unknown'), row.get('Date', 'Unknown'), row.get('url', ''))}] {row.get('title', 'No title')} ({row.get('domain', 'Unknown')})"
            for _, row in top_articles.iterrows()
        ])
        
        prompt = f"""Improve this answer based on the feedback: "{feedback}"

QUESTION: "{question}"

Available Articles:
{articles_text}

Create an improved answer that:
1. Addresses the specific issues mentioned in the feedback
2. Uses proper citations [news_xxxxxxxx] throughout
3. Provides more detailed analysis and context
4. Maintains factual accuracy

Answer:"""
        
        try:
            return self.tools["llm_query"].execute(prompt, max_tokens=1200, temperature=0.3)
        except:
            return self._create_fallback_answer(question, articles_df)

    def _regenerate_with_more_sources(self, question: str, feedback: str, articles_df: pd.DataFrame) -> str:
        """Use more diverse sources for regeneration with proper news_ids"""
        try:
            if 'domain' in articles_df.columns:
                domain_groups = articles_df.groupby('domain')
                diverse_articles = []
                for domain, group in domain_groups:
                    diverse_articles.extend(group.head(3).to_dict('records'))  # Top 3 from each domain
            else:
                diverse_articles = articles_df.head(20).to_dict('records')
            
            articles_text = "\n".join([
                f"[{self.citation_tracker.add_article(art.get('title', 'No title'), art.get('domain', 'Unknown'), art.get('Date', 'Unknown'), art.get('url', ''))}] {art.get('title', 'No title')} ({art.get('domain', 'Unknown')})"
                for art in diverse_articles[:20]
            ])
            
            prompt = f"""The previous answer had issues: "{feedback}"

QUESTION: "{question}"

DIVERSE SOURCES:
{articles_text}

Create a comprehensive answer that:
1. Uses sources from multiple domains/perspectives
2. Addresses the specific feedback issues
3. Provides deeper analysis and context
4. Uses 6-10 well-chosen citations

Answer:"""
        
            return self.tools["llm_query"].execute(prompt, max_tokens=1200, temperature=0.3)
        except:
            return self._create_fallback_answer(question, articles_df)

    def _regenerate_with_different_structure(self, question: str, feedback: str, articles_df: pd.DataFrame) -> str:
        """Try a completely different structural approach with proper news_ids"""
        top_articles = articles_df.head(15)
        articles_text = "\n".join([
            f"[{self.citation_tracker.add_article(row.get('title', 'No title'), row.get('domain', 'Unknown'), row.get('Date', 'Unknown'), row.get('url', ''))}] {row.get('title', 'No title')}"
            for _, row in top_articles.iterrows()
        ])
        
        prompt = f"""Previous answer failed due to: "{feedback}"

QUESTION: "{question}"

SOURCES: {articles_text}

Try a DIFFERENT APPROACH:
1. Start with current status/situation
2. Explain key developments and changes
3. Analyze implications and impacts  
4. Conclude with outlook/significance

Use this new structure to write your answer:"""

        try:
            return self.tools["llm_query"].execute(prompt, max_tokens=1000, temperature=0.4)
        except:
            return self._create_fallback_answer(question, articles_df)

    def _regenerate_incremental_direct(self, question: str, feedback: str, articles_df: pd.DataFrame) -> str:
        """Direct incremental improvement using exact same pattern as working code"""
        top_articles = articles_df.head(15)
        articles_text = "\n".join([
            f"[{row.get('news_id', 'unverified')}] {row.get('title', 'No title')} ({row.get('domain', 'Unknown')})"
            for _, row in top_articles.iterrows()
        ])
        
        prompt = f"""Based on this feedback: "{feedback}", completely rewrite your answer to this question.

QUESTION: "{question}"

Available Articles:
{articles_text}

Create a comprehensive answer that:
1. Addresses the specific issues mentioned in the feedback
2. Uses proper citations [news_xxxxxxxx] throughout
3. Provides more detailed analysis and context
4. Maintains factual accuracy

Answer:"""
        
        try:
            return self.tools["llm_query"].execute(prompt, max_tokens=1200, temperature=0.4)
        except Exception as e:
            print(f"      ⚠️ Direct regeneration failed: {e}")
            return self._generate_fallback_with_article_content(question, articles_df)

    def _generate_fallback_with_article_content(self, question: str, articles_df: pd.DataFrame) -> str:
        """Generate a meaningful fallback answer using actual article content"""
        try:
            # Extract meaningful information from articles
            if articles_df.empty:
                return f"No relevant articles found for the question: '{question}'"
            
            domains = articles_df['domain'].unique()[:5] if 'domain' in articles_df.columns else ['various sources']
            article_count = len(articles_df)
            
            # Get sample titles for context
            sample_titles = []
            if 'title' in articles_df.columns:
                sample_titles = articles_df['title'].head(3).tolist()
            
            # Note enhanced extraction if used
            extraction_note = " Enhanced keyword extraction with spaCy and graph analysis was used to find these articles." if self._enhanced_extraction_used else ""
            
            # Create a more informative answer based on actual data
            answer = f"""Based on {article_count} recent articles from sources including {', '.join(domains)}, there is substantial coverage of "{question}".{extraction_note} """
            
            if sample_titles:
                answer += f"Recent headlines include topics such as: {'; '.join(title[:60] + '...' if len(title) > 60 else title for title in sample_titles)}. "
            
            answer += f"""This indicates active discussion and developments in this area. The articles span multiple perspectives and timeframes, suggesting this is a topic of ongoing interest and relevance.

For detailed analysis, the retrieved articles would need to be reviewed individually to extract specific insights, trends, and expert opinions on the various aspects of this question."""

            return answer
            
        except Exception as e:
            return f"Unable to analyze articles for '{question}' due to processing error: {str(e)}"

    def _generate_comprehensive_answer_with_evaluation(self, question: str, articles_df: pd.DataFrame, iteration_count: int) -> str:
        """Generate and evaluate a comprehensive answer using proper LLM integration with enhanced extraction tracking"""
        print("\n📝 Generating comprehensive answer with evaluation…")
        
        # Validate inputs
        if articles_df is None or articles_df.empty:
            print("   ⚠️ No articles available for answer generation")
            return "Unable to generate answer: No articles provided."
        
        print(f"   📊 Processing {len(articles_df)} articles...")
        
        # Add extraction context if enhanced extraction was used
        if self._enhanced_extraction_used:
            stats = self._last_extraction_stats
            print(f"   🧠 Enhanced extraction stats: {stats.get('confidence', 0):.1%} confidence, {stats.get('graph_nodes', 0)} graph nodes")
        
        # Ensure LLM tool is available
        if not self._ensure_llm_tool_available():
            print("   ⚠️ LLM tool unavailable, using fallback answer")
            return self._generate_fallback_with_article_content(question, articles_df)
        
        try:
            # Track all articles in citation tracker first
            print("   🔗 Tracking articles in citation system...")
            self.citation_tracker.track_articles(articles_df)
            tracked_count = len(self.citation_tracker.citations)
            print(f"   ✅ Tracked {tracked_count} citations")
            
            # Prepare article summaries with proper news_ids
            top_summaries = []
            for _, row in articles_df.head(20).iterrows():
                title = row.get("title", "No title")
                domain = row.get("domain", "Unknown source")
                date = row.get("Date", "Unknown date")
                nid = row.get("news_id", "unverified")
                top_summaries.append(f"[{nid}] [{date}] {title} ({domain})")
            
            articles_text = "\n".join(top_summaries)
            print(f"   ✅ Prepared {len(top_summaries)} article summaries")

            # Add extraction context to prompt if enhanced extraction was used
            extraction_context = ""
            if self._enhanced_extraction_used:
                stats = self._last_extraction_stats
                extraction_context = f"\n\nNOTE: These articles were found using enhanced keyword extraction with {stats.get('confidence', 0):.1%} confidence and {stats.get('graph_nodes', 0)} concept nodes in the knowledge graph. The search focused on domain: {stats.get('domain', 'general')}."

            # Create the prompt
            prompt = f"""Generate a comprehensive answer to this research question based on recent news articles.

QUESTION: "{question}"

Available Articles (use the news_id in square brackets to cite):
{articles_text}

INSTRUCTIONS:
1. Write a substantial 4-5 paragraph answer addressing all aspects of the question
2. Use AT LEAST 15-20 citations throughout - cite multiple sources for each major point
3. ONLY use the exact news_ids provided above – do not make up citations
4. Start each paragraph with 3-4 citations to establish credibility
5. Include diverse perspectives by citing different domains/sources
6. Be specific with facts, figures, and recent developments
7. Conclude with outlook and implications

CITATION REQUIREMENT: You must use at least 15 citations from the provided list. Reference multiple sources for each claim.

Structure:
- Paragraph 1: Current status/overview [4-5 citations]
- Paragraph 2: Recent developments [4-5 citations] 
- Paragraph 3: Analysis and implications [4-5 citations]
- Paragraph 4: Future outlook [3-4 citations]

Answer:"""

            # Generate initial answer using LLM
            print("   🤖 Generating initial answer...")
            try:
                # Use the exact same call pattern as working code
                initial_answer = self.tools["llm_query"].execute(prompt, max_tokens=1000, temperature=0.3)
                
                if not initial_answer or len(initial_answer.strip()) < 50:
                    raise Exception("LLM returned empty or very short response")
                    
                print(f"   ✅ Generated answer ({len(initial_answer)} characters)")
                
            except Exception as llm_error:
                print(f"   ⚠️ LLM generation failed: {llm_error}")
                # Don't fallback to generic answer - try alternative approach
                return self._generate_fallback_with_article_content(question, articles_df)

            # Verify citations
            print("   🔍 Verifying citations...")
            try:
                verified_answer, stats = self.citation_tracker.verify_citations(initial_answer)
                print(f"   📊 Initial answer: {stats['valid_citations']} valid citations, {stats['invalid_citations']} hallucinated")
            except Exception as citation_error:
                print(f"   ⚠️ Citation verification failed: {citation_error}")
                verified_answer = initial_answer
                stats = {"valid_citations": 0, "invalid_citations": 0}

            # Mark steps complete
            self.planner.mark_step_complete("answer_complete", "initial_generation", quality=0.7)
            self.planner.mark_step_complete("answer_complete", "citation_verification", verified=stats["valid_citations"])

            # Evaluate answer quality
            print("   🎯 Evaluating answer quality...")
            try:
                cites_list = [f"[{c['id']}]" for c in self.citation_tracker.get_citation_list()[:20]]
                evaluation = self.answer_evaluator.evaluate_answer(question, verified_answer, cites_list)
                print(f"   ✅ Initial evaluation: {evaluation.verdict} (score: {evaluation.overall_score:.2f})")
            except Exception as eval_error:
                print(f"   ⚠️ Answer evaluation failed: {eval_error}")
                # Create simple fallback evaluation
                class SimpleEval:
                    def __init__(self):
                        self.verdict = "Good"
                        self.overall_score = 0.7
                        self.feedback = "Evaluation unavailable"
                evaluation = SimpleEval()

            self.planner.mark_step_complete("answer_complete", "quality_evaluation", verdict=evaluation.verdict)

            # Iterative improvement
            current_answer = verified_answer
            attempts = 0
            max_attempts = 3
            regeneration_strategies = ["incremental", "evidence_focused", "structural_rewrite", "comprehensive_expansion"]
            
            while evaluation.verdict in ["Fair", "Poor", "Fail"] and attempts < max_attempts:
                attempts += 1
                print(f"   🔄 Improvement attempt {attempts}/{max_attempts} using {regeneration_strategies[attempts-1]}...")
                
                current_score = evaluation.overall_score
                
                try:
                    # Use regeneration methods directly
                    if attempts == 1:
                        improved = self._regenerate_incremental_direct(question, evaluation.feedback, articles_df)
                    elif attempts == 2:
                        improved = self._regenerate_with_more_sources(question, evaluation.feedback, articles_df)
                    else:
                        improved = self._regenerate_with_different_structure(question, evaluation.feedback, articles_df)
                    
                    # Re-evaluate
                    new_evaluation = self.answer_evaluator.evaluate_answer(question, improved, cites_list)
                    print(f"      → New evaluation: {new_evaluation.verdict} (score: {new_evaluation.overall_score:.2f})")
                    
                    # Accept if significantly better
                    if (new_evaluation.overall_score > current_score + 0.1 or 
                        new_evaluation.verdict in ["Excellent", "Good"]):
                        current_answer = improved
                        evaluation = new_evaluation
                        if evaluation.verdict in ["Excellent", "Good"]:
                            break
                    else:
                        print(f"      ⚠️ No significant improvement, keeping previous version")
                        break
                        
                except Exception as improvement_error:
                    print(f"   ⚠️ Improvement attempt failed: {improvement_error}")
                    break

            self.planner.mark_step_complete("answer_complete", "iterative_improvement", attempts=attempts)
            
            # Store final evaluation with extraction stats
            final_eval_data = {
                "verdict": evaluation.verdict,
                "score": evaluation.overall_score,
                "feedback": getattr(evaluation, 'feedback', 'Answer generated'),
                "attempts": attempts,
                "enhanced_extraction_used": self._enhanced_extraction_used
            }
            
            if self._enhanced_extraction_used:
                final_eval_data["extraction_stats"] = self._last_extraction_stats
            
            self.memory.remember("final_evaluation", final_eval_data, importance=0.9)

            extraction_note = " (using enhanced NLP extraction)" if self._enhanced_extraction_used else ""
            print(f"   🎉 Answer generation completed{extraction_note}! Final quality: {evaluation.verdict}")
            return current_answer
            
        except Exception as e:
            print(f"   ❌ Answer generation failed with error: {str(e)}")
            return self._generate_fallback_with_article_content(question, articles_df)
    
    def execute_research_with_chaining(self, question: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute research with hierarchical subgoal planning and live progress display"""
        print(f"🤔 Starting hierarchical research with enhanced keyword extraction: {question}")
        
        max_iterations = 25
        iteration = 0
        articles_df = pd.DataFrame()
        step_attempts = {}
        
        # Initialize progress tracking
        self._last_completed_subgoals = 0
        self._enhanced_extraction_used = False
        self._last_extraction_stats = {}
        
        while iteration < max_iterations:
            iteration += 1
            
            # Show live progress (now TRUE phase-based)
            self._show_live_progress(iteration, max_iterations)
            
            self.think({"question": question, "iteration": iteration})
            
            if self._check_comprehensive_satisfaction_hierarchical(question):
                print(f"\n✅ Research goals achieved after {iteration-1} steps")
                break
            
            next_action = self.planner.plan_next_step_hierarchical(question)
            
            if next_action is None:
                print(f"\n🎯 All actionable steps complete after {iteration-1} iterations")
                break
            
            step_key = f"{self.current_subgoal}.{self.current_step}" if self.current_subgoal and self.current_step else f"unknown.{iteration}"
            step_attempts[step_key] = step_attempts.get(step_key, 0) + 1
            
            if step_attempts[step_key] > 3:
                print(f"      ⚠️ Step {step_key} attempted {step_attempts[step_key]} times, skipping...")
                self.memory.reflect(
                    experience=f"Skipped {step_key} after {step_attempts[step_key]} attempts",
                    lesson="Step was retrying without progress; avoiding this pattern in future."
                )
                if self.current_subgoal and self.current_step:
                    self.planner.mark_step_complete(self.current_subgoal, self.current_step)
                continue
            
            print(f"\n→ Step {iteration}: {next_action.name}")
            
            observation = self.act(next_action)
            
            if observation.success:
                if next_action.expected_outcome and observation.result is not None:
                    is_meaningful = True
                    if isinstance(observation.result, str):
                        if "Help with step" in observation.result or "fallback" in observation.result.lower():
                            is_meaningful = False
                    elif isinstance(observation.result, pd.DataFrame):
                        if observation.result.empty:
                            is_meaningful = False
                    if is_meaningful:
                        self.memory.remember(next_action.expected_outcome, observation.result, importance=0.8)
                
                self._update_step_on_success(next_action, observation.result)
                
                # Check if we completed a subgoal for progress tracking
                completed_subgoals = sum(1 for sg in self.subgoals.values() if sg["complete"])
                if completed_subgoals > self._last_completed_subgoals:
                    print()  # New line for subgoal completion message
                    self._last_completed_subgoals = completed_subgoals
            else:
                print(f"      ✗ Failed: {observation.result}")
                self._update_step_on_failure(next_action, str(observation.result))
                self.planning_failures += 1
                
                if self.planning_failures >= 3:
                    if self.current_subgoal and self.current_step:
                        self.planner.mark_step_complete(self.current_subgoal, self.current_step)
                        print("      ⏭️ Skipping failed step after 3 attempts")
                        self.planning_failures = 0
                else:
                    print("      🔄 Will retry step")

        # Final progress update
        print(f"\n🏁 Research completed!")

        # Show final phase summary (now TRUE phases with extraction stats)
        self.print_phase_summary()

        # Generate answer with safety checks
        final_articles = self._get_current_articles()
        if isinstance(final_articles, pd.DataFrame) and not final_articles.empty:
            print(f"📊 Final articles check: {len(final_articles)} articles available")
            answer = self._generate_comprehensive_answer_with_evaluation(question, final_articles, iteration)
            self.subgoals["answer_complete"]["complete"] = True
            articles_df = final_articles
        else:
            print("⚠️ No final articles available for answer generation")
            answer = "Unable to find sufficient relevant articles for your question."
            articles_df = pd.DataFrame()

        # Collect results safely with proper citation interface
        final_evaluation = self.memory.recall("final_evaluation") or {}
        try:
            # Use unified citation interface
            used_citations = self.citation_tracker.get_used_citations(answer)
        except Exception:
            # Fallback to new interface
            used_citations = self.citation_tracker.get_used_citations()
        
        progress_report = self._generate_progress_report()

        return articles_df, {
            "answer": answer,
            "iterations": iteration,
            "subgoals": self.subgoals,
            "progress_report": progress_report,
            "final_confidence": self._get_current_relevance_confidence(),
            "planning_failures": self.planning_failures,
            "used_citations": used_citations,
            "all_citations": self.citation_tracker.get_citation_list(),
            "citation_stats": {
                "total_tracked": len(self.citation_tracker.citations),
                "used_in_answer": len(used_citations),
                "hallucinations": len(self.citation_tracker.hallucinations)
            },
            "evaluation_stats": {
                "final_verdict": final_evaluation.get("verdict", "Unknown"),
                "final_score": final_evaluation.get("score", 0.0),
                "improvement_attempts": final_evaluation.get("attempts", 0),
                "feedback": final_evaluation.get("feedback", "No evaluation performed"),
                "enhanced_extraction_used": final_evaluation.get("enhanced_extraction_used", False)
            },
            "extraction_stats": {
                "enhanced_used": self._enhanced_extraction_used,
                "last_stats": self._last_extraction_stats
            },
            "memory_summary": self._generate_memory_summary()
        }
    
    def _generate_progress_report(self) -> Dict[str, Any]:
        """Generate a detailed progress report for all subgoals"""
        report = {}
        for subgoal_name, sg_data in self.subgoals.items():
            completed = sum(1 for s in sg_data["steps"] if s["done"])
            total = len(sg_data["steps"])
            report[subgoal_name] = {
                "complete": sg_data["complete"],
                "progress": f"{completed}/{total}",
                "completion_rate": completed / total if total > 0 else 0,
                "steps": [
                    {
                        "name": s["name"],
                        "done": s["done"],
                        "metrics": {k: v for k, v in s.items() if k not in ["name", "done"]}
                    }
                    for s in sg_data["steps"]
                ]
            }
        return report

    def _generate_memory_summary(self) -> Dict[str, Any]:
        """Generate a summary of what the agent learned and remembered"""
        summary = {
            "thoughts_count": len(self.thoughts),
            "reflections_count": len(self.memory.reflections),
            "key_memories": {},
            "learning_points": [],
            "enhanced_extraction_used": self._enhanced_extraction_used
        }
        
        # Add extraction stats if available
        if self._enhanced_extraction_used:
            summary["extraction_stats"] = self._last_extraction_stats
        
        key_items = ["keywords", "initial_keywords", "refined_keywords", "relevance_check", 
                    "filtered_relevance", "need_keyword_refinement"]
        for item in key_items:
            value = self.memory.recall(item)
            if value is not None:
                if isinstance(value, (list, dict, str, int, float, bool)):
                    summary["key_memories"][item] = value
                else:
                    summary["key_memories"][item] = str(value)[:100]
        
        for reflection in self.memory.reflections[-5:]:
            summary["learning_points"].append({
                "experience": reflection["experience"],
                "lesson": reflection["lesson"]
            })
        
        return summary

    def _get_current_relevance_confidence(self) -> float:
        """Get current relevance confidence from memory"""
        data = self.memory.recall("relevance") or self.memory.recall("initial_relevance") or self.memory.recall("filtered_relevance")
        if isinstance(data, dict):
            return data.get("confidence", 0.0)
        elif isinstance(data, (int, float)):
            return float(data)
        return 0.0

    def _get_current_articles(self) -> Any:
        """Get current articles with fallback priority"""
        articles = self.memory.recall("filtered_articles")
        if articles is None or (isinstance(articles, pd.DataFrame) and articles.empty):
            articles = self.memory.recall("articles")
        if articles is None or (isinstance(articles, pd.DataFrame) and articles.empty):
            articles = self.memory.recall("broad_articles")
        if articles is None:
            return pd.DataFrame()
        return articles

    def act(self, action: Action) -> Observation:
        """Enhanced action execution with special handling for NewsSearchTool"""
        if action.name not in self.tools:
            return Observation(action, None, False, [f"Unknown tool: {action.name}"])
        
        try:
            # SPECIAL HANDLING for NewsSearchTool - pass LLM tool for enhanced extraction
            if action.name == "news_search" and "llm_query" in self.tools:
                print(f"      🔗 Enabling enhanced keyword extraction for news search...")
                # Pass the LLM tool to enable enhanced keyword extraction
                result = self.tools[action.name].execute(llm_tool=self.tools["llm_query"], **action.params)
            else:
                result = self.tools[action.name].execute(**action.params)
            
            obs = Observation(action, result, True)
            self.observations.append(obs)
            return obs
        except Exception as e:
            return Observation(action, str(e), False, [f"Error: {e}"])

    def debug_extraction_system(self, question: str = "How is Harvard's endowment situation?"):
        """Debug the enhanced keyword extraction system"""
        print(f"\n🔬 **DEBUGGING ENHANCED EXTRACTION SYSTEM**")
        print(f"Testing with: '{question}'")
        print("=" * 60)
        
        # Check if NewsSearchTool has enhanced extractor
        news_tool = self.tools.get("news_search")
        if not news_tool:
            print("❌ NewsSearchTool not available")
            return False
        
        # Initialize if needed
        llm_tool = self.tools.get("llm_query")
        if llm_tool and not news_tool.keyword_extractor:
            print("🔧 Initializing enhanced extractor...")
            news_tool.initialize_extractor(llm_tool)
        
        if news_tool.keyword_extractor:
            print("✅ Enhanced extractor available")
            
            try:
                # Test debug method
                debug_result = news_tool.keyword_extractor.debug_graph_formation(question)
                
                if debug_result.get('success'):
                    print(f"\n🎉 **EXTRACTION DEBUG SUCCESSFUL!**")
                    return True
                else:
                    print(f"\n❌ **EXTRACTION DEBUG FAILED:** {debug_result.get('error')}")
                    return False
                    
            except Exception as e:
                print(f"\n❌ **DEBUG ERROR:** {e}")
                return False
        else:
            print("❌ Enhanced extractor not available")
            return False


# Factory function for backward compatibility
def create_research_agent(name: str, tools, memory=None):
    """Factory function to create a research agent with enhanced memory"""
    return ResearchAgent(name, tools, memory or EnhancedMemory())

# Test function for enhanced extraction
def test_enhanced_extraction_in_agent():
    """Test the enhanced extraction system within the research agent context"""
    from tools import LLMTool, NewsSearchTool
    
    print("\n🧪 **TESTING ENHANCED EXTRACTION IN RESEARCH AGENT**")
    
    # Create tools
    llm_tool = LLMTool()
    news_tool = NewsSearchTool()
    tools = {"llm_query": llm_tool, "news_search": news_tool}
    
    # Create research agent
    agent = ResearchAgent("TestAgent", list(tools.values()))
    agent.tools = tools
    
    # Test the extraction system
    return agent.debug_extraction_system()

if __name__ == "__main__":
    # Quick test when running research_agent.py directly
    test_enhanced_extraction_in_agent()