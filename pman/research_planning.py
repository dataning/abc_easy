# research_planning.py - Planning and step management for research agents

import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from memory import Action

class ResearchPlanner:
    """Handles hierarchical planning and step management for research agents"""
    
    def __init__(self, agent):
        self.agent = agent
        self.subgoals = self._initialize_subgoals()
        
    def _initialize_subgoals(self) -> Dict[str, Dict]:
        """Initialize the hierarchical subgoal structure with phases"""
        return {
            "clarity_checked": {
                "phase": "🔍 Phase 1: Question Analysis",
                "phase_summary": "Analyzing question clarity and scope",
                "steps": [{"name": "question_analysis", "done": False, "confidence": 0.0}],
                "complete": False
            },
            "keywords_extracted": {
                "phase": "🔍 Phase 2: Keyword Extraction", 
                "phase_summary": "Extracting and refining search keywords",
                "steps": [
                    {"name": "initial_extraction", "done": False, "quality": 0.0},
                    {"name": "keyword_refinement", "done": False, "quality": 0.0}
                ],
                "complete": False
            },
            "articles_fetched": {
                "phase": "🔍 Phase 3: Article Retrieval",
                "phase_summary": "Fetching relevant articles from multiple sources",
                "steps": [
                    {"name": "broad_fetch", "done": False, "count": 0},
                    {"name": "focused_fetch", "done": False, "count": 0},
                    {"name": "domain_diversification", "done": False, "diversity": 0.0}
                ],
                "complete": False,
                "min_articles": 20,
                "target_articles": 50
            },
            "sufficient_relevance": {
                "phase": "🔍 Phase 4: Relevance Filtering",
                "phase_summary": "Evaluating and improving article relevance",
                "steps": [
                    {"name": "initial_relevance_check", "done": False, "confidence": 0.0},
                    {"name": "refine_keywords_if_low", "done": False, "confidence": 0.0},
                    {"name": "semantic_filtering", "done": False, "confidence": 0.0}
                ],
                "complete": False,
                "threshold": 0.4,
                "target_threshold": 0.6
            },
            "domain_diversity": {
                "phase": "🔍 Phase 5: Domain Diversification",
                "phase_summary": "Ensuring diverse source coverage",
                "steps": [
                    {"name": "diversity_assessment", "done": False, "score": 0.0},
                    {"name": "gap_filling_search", "done": False, "score": 0.0}
                ],
                "complete": False,
                "min_domains": 3,
                "target": 0.6
            },
            "answer_complete": {
                "phase": "🔍 Phase 6: Answer Generation",
                "phase_summary": "Generating and evaluating comprehensive answer",
                "steps": [
                    {"name": "initial_generation", "done": False, "quality": 0.0},
                    {"name": "citation_verification", "done": False, "verified": 0},
                    {"name": "quality_evaluation", "done": False, "verdict": "Unknown"},
                    {"name": "iterative_improvement", "done": False, "attempts": 0}
                ],
                "complete": False,
                "min_quality": 0.7
            }
        }

    def get_phase_summary(self) -> Dict[str, str]:
        """Get a high-level phase summary for display"""
        phases = {}
        for subgoal_name, subgoal_data in self.subgoals.items():
            phase_name = subgoal_data.get("phase", f"Phase: {subgoal_name}")
            completed_steps = sum(1 for step in subgoal_data["steps"] if step["done"])
            total_steps = len(subgoal_data["steps"])
            
            if subgoal_data["complete"]:
                status = "✅ Done"
                detail = self._get_phase_completion_detail(subgoal_name, subgoal_data)
            elif completed_steps > 0:
                status = f"🔄 {completed_steps}/{total_steps}"
                detail = self._get_phase_progress_detail(subgoal_name, subgoal_data)
            else:
                status = "⏳ Pending"
                detail = ""
            
            phases[phase_name] = f"{status}{' → ' + detail if detail else ''}"
        
        return phases

    def _get_phase_completion_detail(self, subgoal_name: str, subgoal_data: Dict) -> str:
        """Get completion details for a finished phase"""
        if subgoal_name == "keywords_extracted":
            keywords = self.agent.memory.recall("keywords") or []
            if isinstance(keywords, list) and len(keywords) > 0:
                return f"(Keywords: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''})"
            return "(Keywords extracted)"
        elif subgoal_name == "articles_fetched":
            articles = self.agent._get_current_articles()
            count = len(articles) if isinstance(articles, pd.DataFrame) else 0
            return f"({count} articles retrieved)"
        elif subgoal_name == "sufficient_relevance":
            confidence = self.agent._get_current_relevance_confidence()
            articles = self.agent._get_current_articles()
            relevant_count = int(len(articles) * confidence) if isinstance(articles, pd.DataFrame) else 0
            total_count = len(articles) if isinstance(articles, pd.DataFrame) else 0
            return f"({relevant_count} of {total_count} ({confidence:.1%}) passed threshold)"
        elif subgoal_name == "answer_complete":
            eval_data = self.agent.memory.recall("final_evaluation") or {}
            verdict = eval_data.get("verdict", "Unknown")
            return f"(Quality: {verdict})"
        elif subgoal_name == "clarity_checked":
            return "(Question analyzed)"
        elif subgoal_name == "domain_diversity":
            articles = self.agent._get_current_articles()
            if isinstance(articles, pd.DataFrame) and 'domain' in articles.columns:
                unique_domains = articles['domain'].nunique()
                return f"({unique_domains} unique domains)"
            return "(Diversity assessed)"
        return ""

    def _get_phase_progress_detail(self, subgoal_name: str, subgoal_data: Dict) -> str:
        """Get progress details for an in-progress phase"""
        completed_steps = [step for step in subgoal_data["steps"] if step["done"]]
        if not completed_steps:
            return ""
        
        latest_step = completed_steps[-1]
        if subgoal_name == "articles_fetched" and "count" in latest_step:
            return f"({latest_step['count']} articles so far)"
        elif subgoal_name == "sufficient_relevance" and "confidence" in latest_step:
            return f"({latest_step['confidence']:.1%} relevance)"
        elif subgoal_name == "keywords_extracted":
            return "(Keywords in progress)"
        elif subgoal_name == "answer_complete":
            return "(Answer generation in progress)"
        return ""

    def mark_step_complete(self, subgoal_name: str, step_name: str, **metrics):
        """Mark a step as complete and update metrics"""
        sg = self.subgoals[subgoal_name]
        for step in sg["steps"]:
            if step["name"] == step_name:
                step["done"] = True
                for k, v in metrics.items():
                    if k in step:
                        step[k] = v
                break
        
        if all(step["done"] for step in sg["steps"]):
            sg["complete"] = True
            phase_name = sg.get("phase", f"Phase: {subgoal_name}")
            print(f"      ✅ {phase_name} completed!")
            
        if self.agent.current_subgoal == subgoal_name and self.agent.current_step == step_name:
            self.agent.current_step = None
            self.agent.current_subgoal = None

    def should_skip_step(self, subgoal_name: str, step_name: str, step_data: Dict) -> bool:
        """Determine if a step should be skipped based on current conditions"""
        # Keyword refinement can be skipped if we have high-quality keywords
        if subgoal_name == "keywords_extracted" and step_name == "keyword_refinement":
            init = next(filter(lambda s: s["name"]=="initial_extraction", self.subgoals["keywords_extracted"]["steps"]))
            keywords = self.agent.memory.recall("initial_keywords") or []
            if init.get("quality",0) >= 0.8 and isinstance(keywords, list) and len(set(keywords)) >= 4:
                return True
        
        # Focused fetch can be skipped if we have enough articles from broad fetch
        if subgoal_name == "articles_fetched" and step_name == "focused_fetch":
            broad = next(filter(lambda s: s["name"]=="broad_fetch", self.subgoals["articles_fetched"]["steps"]))
            if broad.get("count",0) >= self.subgoals["articles_fetched"]["target_articles"]:
                return True
        
        # Keyword refinement in relevance can be skipped if relevance is already good
        if subgoal_name == "sufficient_relevance" and step_name == "refine_keywords_if_low":
            ir = next(filter(lambda s: s["name"]=="initial_relevance_check", self.subgoals["sufficient_relevance"]["steps"]))
            if ir.get("confidence",0) >= self.subgoals["sufficient_relevance"]["target_threshold"]:
                return True
        
        # Gap filling can be skipped if diversity is already sufficient
        if subgoal_name == "domain_diversity" and step_name == "gap_filling_search":
            da = next(filter(lambda s: s["name"]=="diversity_assessment", self.subgoals["domain_diversity"]["steps"]))
            if da.get("score",0) >= self.subgoals["domain_diversity"]["target"]:
                return True
        
        return False

    def get_next_incomplete_step(self) -> Optional[Tuple[str, Dict, str, Dict]]:
        """Find the next incomplete step in the hierarchical plan"""
        for sg_name, sg_data in self.subgoals.items():
            if sg_data["complete"]:
                continue
            for step in sg_data["steps"]:
                if not step["done"]:
                    return sg_name, sg_data, step["name"], step
        return None

    def plan_next_step_hierarchical(self, question: str) -> Optional[Action]:
        """Plan next step with better loop detection"""
        
        # Check for repeated failures
        step_key = f"{self.agent.current_subgoal}.{self.agent.current_step}"
        failure_count = getattr(self, '_step_failures', {}).get(step_key, 0)
        
        if failure_count >= 3:
            print(f"      ⚠️ Step {step_key} failed {failure_count} times, marking complete and moving on")
            if self.agent.current_subgoal and self.agent.current_step:
                self.mark_step_complete(self.agent.current_subgoal, self.agent.current_step)
            return None
        
        # Clear current step if it's been completed
        if (self.agent.current_subgoal and self.agent.current_step and 
            self.agent.current_subgoal in self.subgoals):
            current_sg = self.subgoals[self.agent.current_subgoal]
            current_step_obj = next(
                (s for s in current_sg["steps"] if s["name"] == self.agent.current_step), None
            )
            if current_step_obj and current_step_obj["done"]:
                print(f"      ✅ Step {self.agent.current_step} completed, advancing...")
                self.agent.current_step = None
                self.agent.current_subgoal = None

        # Auto-complete keyword refinement if conditions met
        ke = self.subgoals["keywords_extracted"]
        init_step = next((s for s in ke["steps"] if s["name"]=="initial_extraction"), None)
        if init_step and init_step["done"]:
            keywords = self.agent.memory.recall("initial_keywords") or []
            if (init_step.get("quality", 0) >= 0.8 and 
                isinstance(keywords, list) and 
                len(set(keywords)) >= 4 and 
                not ke["complete"]):
                kr = next((s for s in ke["steps"] if s["name"]=="keyword_refinement"), None)
                if kr and not kr["done"]:
                    kr["done"] = True
                    kr["quality"] = init_step["quality"]
                    ke["complete"] = True
                    print("      ✅ Phase 2: Keyword Extraction auto-completed (≥4 high-quality keywords)")

        # Get current state
        articles = self.agent._get_current_articles()
        articles_count = len(articles) if isinstance(articles, pd.DataFrame) else 0
        relevance = self.agent._get_current_relevance_confidence()

        # Priority logic for early article collection
        if articles_count >= self.subgoals["articles_fetched"]["min_articles"]:
            # Check relevance first if we have enough articles
            ir_step = next(
                (s for s in self.subgoals["sufficient_relevance"]["steps"]
                 if s["name"] == "initial_relevance_check"), None
            )
            if ir_step and not ir_step["done"]:
                self.agent.current_subgoal = "sufficient_relevance"
                self.agent.current_step = "initial_relevance_check"
                print(f"      🎯 Working on: {self.agent.current_subgoal}.{self.agent.current_step}")
                return self.create_action_for_step(
                    question, "sufficient_relevance", "initial_relevance_check", 
                    ir_step, self.subgoals["sufficient_relevance"]
                )

            # If relevance is low, try semantic filtering
            if relevance < self.subgoals["sufficient_relevance"]["threshold"]:
                sf_step = next(
                    (s for s in self.subgoals["sufficient_relevance"]["steps"]
                     if s["name"] == "semantic_filtering"), None
                )
                if sf_step and not sf_step["done"]:
                    self.agent.current_subgoal = "sufficient_relevance"
                    self.agent.current_step = "semantic_filtering"
                    print(f"      🎯 Working on: {self.agent.current_subgoal}.{self.agent.current_step}")
                    return self.create_action_for_step(
                        question, "sufficient_relevance", "semantic_filtering", 
                        sf_step, self.subgoals["sufficient_relevance"]
                    )

            # If relevance is sufficient, move to answer generation
            if relevance >= self.subgoals["sufficient_relevance"]["threshold"]:
                return self._create_action_for_answer_generation(question)

        # Early stage: need keywords and initial articles
        keywords_extracted = self.agent.memory.recall("keywords") or self.agent.memory.recall("initial_keywords")
        if articles_count < 5:
            if not keywords_extracted:
                return self._create_keyword_extraction_action(question)
            else:
                return self._create_broad_fetch_action(question)

        # Handle low relevance scenarios
        need_refinement = self.agent.memory.recall("need_keyword_refinement")
        if relevance < 0.3 and not need_refinement:
            return self._create_keyword_refinement_action(question)
        elif relevance < self.subgoals["sufficient_relevance"]["threshold"]:
            return self._create_semantic_filtering_action(question)

        # Need more articles
        if articles_count < self.subgoals["articles_fetched"]["min_articles"]:
            return self._create_focused_fetch_action(question)

        # Default: follow hierarchical plan
        next_step_info = self.get_next_incomplete_step()
        if not next_step_info:
            return None
        
        subgoal_name, subgoal_data, step_name, step_data = next_step_info
        
        if self.should_skip_step(subgoal_name, step_name, step_data):
            self.mark_step_complete(subgoal_name, step_name)
            print(f"      ⏭️ Skipping '{step_name}' (conditions met)")
            return self.plan_next_step_hierarchical(question)
        
        self.agent.current_subgoal = subgoal_name
        self.agent.current_step = step_name
        
        print(f"      🎯 Working on: {subgoal_name}.{step_name}")
        
        return self.create_action_for_step(question, subgoal_name, step_name, step_data, subgoal_data)

    def create_action_for_step(self, question: str, subgoal_name: str, step_name: str,
                              step_data: Dict, subgoal_data: Dict) -> Action:
        """Create specific actions for each step type"""
        if subgoal_name == "clarity_checked" and step_name == "question_analysis":
            return Action(
                name="llm_query",
                params={"prompt": f"Analyze this research question for clarity and scope: '{question}'. Rate clarity 1-10 and suggest any improvements.",
                        "max_tokens": 128},
                expected_outcome="clarity_analysis"
            )
        
        if subgoal_name == "keywords_extracted":
            if step_name == "initial_extraction":
                return Action(
                    name="llm_query",
                    params={"prompt": f"Extract 3-5 primary search keywords from: '{question}'. Return only keywords separated by commas.",
                            "max_tokens": 64},
                    expected_outcome="initial_keywords"
                )
            if step_name == "keyword_refinement":
                return Action(
                    name="llm_query",
                    params={"prompt": f"The keywords for '{question}' yielded low relevance. Generate 3-5 alternative/broader keywords. Return only keywords separated by commas.",
                            "max_tokens": 64},
                    expected_outcome="refined_keywords"
                )
        
        if subgoal_name == "articles_fetched":
            kws = self.agent.memory.recall("keywords") or self.agent.memory.recall("refined_keywords") or [question]
            if step_name == "broad_fetch":
                return Action(
                    name="news_search",
                    params={"keywords": kws, "max_records": 250, "citation_tracker": self.agent.citation_tracker},
                    expected_outcome="broad_articles"
                )
            if step_name == "focused_fetch":
                return Action(
                    name="news_search",
                    params={"keywords": self._get_focused_keywords(question, kws),
                            "max_records": 200, "citation_tracker": self.agent.citation_tracker},
                    expected_outcome="focused_articles"
                )
            if step_name == "domain_diversification":
                return Action(
                    name="news_search",
                    params={"keywords": self._get_diversification_keywords(question),
                            "max_records": 150, "citation_tracker": self.agent.citation_tracker},
                    expected_outcome="diverse_articles"
                )
        
        if subgoal_name == "sufficient_relevance":
            if step_name == "initial_relevance_check":
                df = self.agent._get_current_articles()
                kws = self.agent.memory.recall("keywords") or []
                return Action(
                    name="evaluate_relevance",
                    params={"query": question, "articles_df": df, "keywords": kws},
                    expected_outcome="initial_relevance"
                )
            if step_name == "refine_keywords_if_low":
                return self.create_action_for_step(question, "keywords_extracted", "keyword_refinement", {}, {})
            if step_name == "semantic_filtering":
                df = self.agent._get_current_articles()
                return Action(
                    name="semantic_filter",
                    params={"articles_df": df, "query": question, "top_k": 100},
                    expected_outcome="filtered_articles"
                )
        
        if subgoal_name == "domain_diversity":
            df = self.agent._get_current_articles()
            if step_name == "diversity_assessment":
                return Action(
                    name="llm_query",
                    params={"prompt": f"Assess domain diversity of these news sources for question '{question}': {self._get_domain_summary(df)}. Rate diversity 1-10.",
                            "max_tokens": 128},
                    expected_outcome="diversity_assessment"
                )
            if step_name == "gap_filling_search":
                gap = self._identify_domain_gaps(df, question)
                return Action(
                    name="news_search",
                    params={"keywords": gap, "max_records": 100, "citation_tracker": self.agent.citation_tracker},
                    expected_outcome="gap_articles"
                )
        
        # Fallback action
        return Action(
            name="llm_query",
            params={"prompt": f"Help with step {step_name} for {subgoal_name}", "max_tokens": 64},
            expected_outcome="fallback"
        )

    def _create_keyword_extraction_action(self, question: str) -> Action:
        """Create action for initial keyword extraction"""
        self.agent.current_subgoal = "keywords_extracted"
        self.agent.current_step = "initial_extraction"
        return Action(
            name="llm_query",
            params={"prompt": f"Extract 3-5 primary search keywords from: '{question}'. Return only keywords separated by commas.",
                    "max_tokens": 64},
            expected_outcome="initial_keywords"
        )

    def _create_keyword_refinement_action(self, question: str) -> Action:
        """Create action for keyword refinement"""
        self.agent.current_subgoal = "keywords_extracted"
        self.agent.current_step = "keyword_refinement"
        return Action(
            name="llm_query",
            params={"prompt": f"The keywords for '{question}' yielded low relevance. Generate 3-5 alternative/broader keywords. Return only keywords separated by commas.",
                    "max_tokens": 64},
            expected_outcome="refined_keywords"
        )

    def _create_broad_fetch_action(self, question: str) -> Action:
        """Create action for broad article fetching"""
        self.agent.current_subgoal = "articles_fetched"
        self.agent.current_step = "broad_fetch"
        kws = self.agent.memory.recall("keywords") or self.agent.memory.recall("initial_keywords") or [question]
        return Action(
            name="news_search",
            params={"keywords": kws, "max_records": 250, "citation_tracker": self.agent.citation_tracker},
            expected_outcome="broad_articles"
        )

    def _create_focused_fetch_action(self, question: str) -> Action:
        """Create action for focused article fetching"""
        self.agent.current_subgoal = "articles_fetched"
        self.agent.current_step = "focused_fetch"
        kws = self.agent.memory.recall("keywords") or self.agent.memory.recall("initial_keywords") or [question]
        return Action(
            name="news_search",
            params={"keywords": self._get_focused_keywords(question, kws),
                    "max_records": 200, "citation_tracker": self.agent.citation_tracker},
            expected_outcome="focused_articles"
        )

    def _create_semantic_filtering_action(self, question: str) -> Action:
        """Create action for semantic filtering"""
        self.agent.current_subgoal = "sufficient_relevance"
        self.agent.current_step = "semantic_filtering"
        df = self.agent._get_current_articles()
        return Action(
            name="semantic_filter",
            params={"articles_df": df, "query": question, "top_k": 100},
            expected_outcome="filtered_articles"
        )

    def _create_action_for_answer_generation(self, question: str) -> Action:
        """Create action for answer generation phase"""
        answer_steps = self.subgoals["answer_complete"]["steps"]
        for step in answer_steps:
            if not step["done"]:
                self.agent.current_subgoal = "answer_complete"
                self.agent.current_step = step["name"]
                if step["name"] == "initial_generation":
                    self.mark_step_complete("answer_complete", "initial_generation", quality=0.7)
                    continue
                break
        return None

    def _get_focused_keywords(self, question: str, base_keywords: List[str]) -> List[str]:
        """Generate focused keywords based on question domain"""
        if "harvard" in question.lower():
            return base_keywords + ["Harvard University", "Harvard endowment", "Cambridge"]
        elif "university" in question.lower():
            return base_keywords + ["higher education", "academic", "college"]
        else:
            return base_keywords + [kw + "s" for kw in base_keywords if len(kw) > 3]

    def _get_diversification_keywords(self, question: str) -> List[str]:
        """Generate keywords for domain diversification"""
        if "harvard" in question.lower():
            return ["Harvard Crimson", "Boston Globe Harvard", "Financial Times university"]
        elif "university" in question.lower():
            return ["Chronicle Higher Education", "Inside Higher Ed", "academic news"]
        else:
            return ["news analysis", "expert opinion", "industry report"]

    def _get_domain_summary(self, articles_df: pd.DataFrame) -> str:
        """Get summary of domains in articles"""
        if not isinstance(articles_df, pd.DataFrame) or articles_df.empty or "domain" not in articles_df.columns:
            return "No articles"
        counts = articles_df["domain"].value_counts().head(10)
        return ", ".join(f"{d}({n})" for d, n in counts.items())

    def _identify_domain_gaps(self, articles_df: pd.DataFrame, question: str) -> List[str]:
        """Identify gaps in domain coverage"""
        if not isinstance(articles_df, pd.DataFrame) or articles_df.empty:
            return [question]
        domains = set(articles_df.get("domain", []))
        academic = {d for d in domains if any(t in d.lower() for t in ["edu", "harvard", "university", "academic"])}
        newsd = {d for d in domains if any(t in d.lower() for t in ["news", "times", "post", "journal"])}
        gaps = []
        if len(academic) < 2:
            gaps.extend(["official statement", "university press release"])
        if len(newsd) < 3:
            gaps.extend(["breaking news", "latest report"])
        return gaps or [question]

    def track_step_failure(self, step_key: str):
        """Track step failures to prevent infinite loops"""
        if not hasattr(self, '_step_failures'):
            self._step_failures = {}
        self._step_failures[step_key] = self._step_failures.get(step_key, 0) + 1