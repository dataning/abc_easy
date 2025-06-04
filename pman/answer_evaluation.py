import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import the citation tracker
from citation_tracker import CitationTracker

@dataclass
class EvaluationResult:
    """Result of answer evaluation with detailed criteria scores"""
    verdict: str  # "Excellent", "Good", "Fair", "Fail"
    overall_score: float  # 0.0 to 1.0
    feedback: str
    flawed_sentences: List[int] = field(default_factory=list)
    specific_issues: List[str] = field(default_factory=list)
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    priority_improvement: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)

class AnswerEvaluator:
    """Enhanced evaluator with detailed criteria and learning from patterns"""
    def __init__(self, llm_tool):
        self.llm_tool = llm_tool
        self.evaluation_history = []
        self.criteria_weights = {
            "completeness": 0.30,
            "evidence_quality": 0.30,
            "clarity": 0.20,
            "accuracy": 0.20
        }

    def evaluate_answer(self, question: str, answer: str, citations_available: List[str]) -> EvaluationResult:
        sentences = self._split_into_sentences(answer)
        
        # Enhanced evaluation prompt with specific criteria
        prompt = f"""You are an expert research answer evaluator. Evaluate this answer using detailed criteria.

QUESTION: "{question}"

ANSWER TO EVALUATE:
{answer}

AVAILABLE CITATIONS: {citations_available[:15]}

EVALUATION CRITERIA (provide score 0.0-1.0 for each):

1. COMPLETENESS (30% weight): Does the answer fully address all aspects of the question?
   - Covers main topics mentioned in question
   - Provides sufficient depth and detail
   - Addresses potential follow-up questions

2. EVIDENCE QUALITY (30% weight): Are citations relevant, authoritative, and well-used?
   - Uses multiple credible sources
   - Citations directly support claims
   - Balances different perspectives if available
   - Avoids over-relying on single sources

3. CLARITY (20% weight): Is the answer well-structured and easy to understand?
   - Clear organization and flow
   - Good transitions between ideas
   - Appropriate language and tone
   - Logical progression of information

4. ACCURACY (20% weight): Are facts presented correctly and appropriately qualified?
   - Information appears factual
   - Appropriate caveats and limitations noted
   - Avoids speculation without evidence
   - Dates and figures used correctly

INSTRUCTIONS:
Return your evaluation as JSON with this exact structure:
{{
    "completeness_score": 0.75,
    "evidence_quality_score": 0.65,
    "clarity_score": 0.80,
    "accuracy_score": 0.70,
    "overall_score": 0.72,
    "verdict": "Good",
    "priority_improvement": "evidence_quality",
    "specific_issues": [
        "Could use more diverse sources",
        "Some claims lack direct citation support"
    ],
    "improvement_suggestions": [
        "Add citations from academic or official sources",
        "Balance perspectives from different domains"
    ],
    "flawed_sentences": [3, 7],
    "detailed_feedback": "The answer covers the main topic but could benefit from stronger evidence..."
}}

SENTENCES FOR REFERENCE:
{self._number_sentences(sentences)}

Evaluation JSON:"""

        try:
            response = self.llm_tool.execute(prompt, max_tokens=1000, temperature=0.1)
            data = self._parse_enhanced_evaluation(response)
            
            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(data)
            data["overall_score"] = overall_score
            
            # Determine verdict based on score
            verdict = self._score_to_verdict(overall_score)
            data["verdict"] = verdict
            
            result = EvaluationResult(
                verdict=data.get("verdict", "Fair"),
                overall_score=data.get("overall_score", 0.5),
                feedback=data.get("detailed_feedback", "No detailed feedback provided"),
                flawed_sentences=data.get("flawed_sentences", []),
                specific_issues=data.get("specific_issues", []),
                criteria_scores={
                    "completeness": data.get("completeness_score", 0.5),
                    "evidence_quality": data.get("evidence_quality_score", 0.5),
                    "clarity": data.get("clarity_score", 0.5),
                    "accuracy": data.get("accuracy_score", 0.5)
                },
                priority_improvement=data.get("priority_improvement", "overall_quality"),
                improvement_suggestions=data.get("improvement_suggestions", [])
            )
            
            # Store evaluation for learning
            self.evaluation_history.append({
                "question": question,
                "result": result,
                "timestamp": datetime.now()
            })
            
            return result
            
        except Exception as e:
            print(f"   ⚠️ Evaluation error: {e}")
            return EvaluationResult(
                verdict="Fair",
                overall_score=0.5,
                feedback=f"Evaluation failed due to error: {e}",
                flawed_sentences=[],
                specific_issues=["Evaluation system error"],
                criteria_scores={"completeness": 0.5, "evidence_quality": 0.5, "clarity": 0.5, "accuracy": 0.5},
                priority_improvement="system_reliability"
            )

    def _calculate_weighted_score(self, data: Dict[str, Any]) -> float:
        """Calculate weighted overall score from criteria scores"""
        total_score = 0.0
        for criterion, weight in self.criteria_weights.items():
            score_key = f"{criterion}_score"
            if criterion == "evidence_quality":
                score_key = "evidence_quality_score"
            score = data.get(score_key, 0.5)
            total_score += score * weight
        return min(max(total_score, 0.0), 1.0)

    def _score_to_verdict(self, score: float) -> str:
        """Convert numeric score to verdict category"""
        if score >= 0.85:
            return "Excellent"
        elif score >= 0.70:
            return "Good"
        elif score >= 0.50:
            return "Fair"
        else:
            return "Fail"

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _number_sentences(self, sentences: List[str]) -> str:
        return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))

    def _parse_enhanced_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse enhanced evaluation response with multiple fallback strategies"""
        try:
            # Try to extract JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return self._fallback_parse_enhanced(response)
        except json.JSONDecodeError:
            return self._fallback_parse_enhanced(response)

    def _fallback_parse_enhanced(self, response: str) -> Dict[str, Any]:
        """Enhanced fallback parsing for evaluation responses"""
        data = {
            "completeness_score": 0.5,
            "evidence_quality_score": 0.5,
            "clarity_score": 0.5,
            "accuracy_score": 0.5,
            "overall_score": 0.5,
            "verdict": "Fair",
            "priority_improvement": "overall_quality",
            "specific_issues": [],
            "improvement_suggestions": [],
            "flawed_sentences": [],
            "detailed_feedback": response[:500]
        }
        
        # Extract verdict
        verdict_match = re.search(r'(Excellent|Good|Fair|Fail)', response, re.IGNORECASE)
        if verdict_match:
            data["verdict"] = verdict_match.group(1).title()
        
        # Extract scores
        score_patterns = [
            (r'completeness[_\s]*score[:\s]*(\d+\.?\d*)', "completeness_score"),
            (r'evidence[_\s]*quality[_\s]*score[:\s]*(\d+\.?\d*)', "evidence_quality_score"),
            (r'clarity[_\s]*score[:\s]*(\d+\.?\d*)', "clarity_score"),
            (r'accuracy[_\s]*score[:\s]*(\d+\.?\d*)', "accuracy_score"),
            (r'overall[_\s]*score[:\s]*(\d+\.?\d*)', "overall_score")
        ]
        
        for pattern, key in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                if score > 1.0:
                    score /= 100.0
                data[key] = min(max(score, 0.0), 1.0)
        
        # Extract sentence numbers
        sentence_matches = re.findall(r'sentence[s]?\s*(\d+)', response, re.IGNORECASE)
        if sentence_matches:
            data["flawed_sentences"] = [int(m) for m in sentence_matches[:5]]
        
        return data

    def get_evaluation_trends(self) -> Dict[str, Any]:
        """Analyze evaluation history to identify patterns"""
        if not self.evaluation_history:
            return {"message": "No evaluation history available"}
        
        recent = self.evaluation_history[-10:]
        avg_score = sum(e["result"].overall_score for e in recent) / len(recent)
        common_issues = {}
        
        for eval_data in recent:
            for issue in eval_data["result"].specific_issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        return {
            "average_recent_score": avg_score,
            "total_evaluations": len(self.evaluation_history),
            "common_issues": sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:5],
            "improvement_trend": "improving" if len(recent) >= 3 and recent[-1]["result"].overall_score > recent[0]["result"].overall_score else "stable"
        }

class AnswerRegenerator:
    """Enhanced regenerator with multiple strategies and better source utilization"""
    def __init__(self, llm_tool):
        self.llm_tool = llm_tool
        self.strategies = [
            "incremental_improvement",
            "evidence_focused",
            "structural_rewrite", 
            "comprehensive_expansion",
            "perspective_shift"
        ]
        self.strategy_history = {}

    def regenerate_answer(self, question: str, original_answer: str, evaluation: EvaluationResult,
                         articles_df, citation_tracker: CitationTracker, strategy: str = None) -> str:
        """Main regeneration method with strategy selection"""
        
        if strategy is None:
            strategy = self._select_optimal_strategy(evaluation)
        
        print(f"      🔄 Using strategy: {strategy}")
        
        # Track strategy usage
        self.strategy_history[strategy] = self.strategy_history.get(strategy, 0) + 1
        
        if strategy == "incremental_improvement":
            return self._incremental_improvement(question, original_answer, evaluation, articles_df, citation_tracker)
        elif strategy == "evidence_focused":
            return self._evidence_focused_rewrite(question, evaluation, articles_df, citation_tracker)
        elif strategy == "structural_rewrite":
            return self._structural_rewrite(question, evaluation, articles_df, citation_tracker)
        elif strategy == "comprehensive_expansion":
            return self._comprehensive_expansion(question, evaluation, articles_df, citation_tracker)
        elif strategy == "perspective_shift":
            return self._perspective_shift_rewrite(question, evaluation, articles_df, citation_tracker)
        else:
            return self._incremental_improvement(question, original_answer, evaluation, articles_df, citation_tracker)

    def _select_optimal_strategy(self, evaluation: EvaluationResult) -> str:
        """Select the best strategy based on evaluation results"""
        priority = evaluation.priority_improvement.lower()
        criteria_scores = evaluation.criteria_scores
        
        # Strategy selection logic based on weakest areas
        if "evidence" in priority or criteria_scores.get("evidence_quality", 0.5) < 0.5:
            return "evidence_focused"
        elif "clarity" in priority or criteria_scores.get("clarity", 0.5) < 0.5:
            return "structural_rewrite"
        elif "completeness" in priority or criteria_scores.get("completeness", 0.5) < 0.5:
            return "comprehensive_expansion"
        elif evaluation.overall_score < 0.4:
            return "perspective_shift"
        else:
            return "incremental_improvement"

    # Include all the regeneration strategy methods here...
    # (I'll include a couple key ones for brevity)

    def _evidence_focused_rewrite(self, question: str, evaluation: EvaluationResult,
                                 articles_df, citation_tracker: CitationTracker) -> str:
        """Focus on stronger evidence and better citation usage"""
        
        # Prioritize recent and authoritative sources
        prioritized_articles = self._prioritize_sources(articles_df, citation_tracker)
        articles_text = "\n".join([
            f"[{art['id']}] {art['title']} - {art['date']} ({art['domain']})"
            for art in prioritized_articles[:20]
        ])
        
        prompt = f"""Rewrite this answer with STRONG EVIDENCE and COMPREHENSIVE CITATIONS.

QUESTION: "{question}"

CURRENT EVALUATION ISSUES:
{evaluation.feedback}

SPECIFIC PROBLEMS TO ADDRESS:
{'; '.join(evaluation.specific_issues)}

HIGH-PRIORITY SOURCES (use extensively):
{articles_text}

EVIDENCE-FOCUSED REQUIREMENTS:
1. Use 10-15 citations from the sources above
2. Each major claim should have 2-3 supporting citations
3. Prioritize recent articles and credible domains (.edu, major news outlets)
4. Balance different perspectives when available
5. Quote or reference specific information from sources
6. Group related sources to build stronger evidence

Write a thoroughly evidenced answer that addresses all evaluation concerns:"""

        try:
            result = self.llm_tool.execute(prompt, max_tokens=1200, temperature=0.2)
            verified_result, stats = citation_tracker.verify_citations(result)
            print(f"         📊 Evidence rewrite: {stats.get('valid_citations', 0)} citations")
            return verified_result
        except Exception as e:
            print(f"         ⚠️ Evidence rewrite failed: {e}")
            return self._fallback_regeneration(question, evaluation, articles_df, citation_tracker)

    # Include other strategy methods...
    # (All the rest of your regeneration methods go here)

    def _prioritize_sources(self, articles_df, citation_tracker: CitationTracker) -> List[Dict[str, str]]:
        """Prioritize sources based on recency, domain authority, and relevance"""
        if not hasattr(articles_df, 'iterrows'):
            return []
        
        articles = []
        for _, row in articles_df.iterrows():
            domain = row.get('domain', '')
            
            # Calculate priority score
            priority_score = 0
            
            # Domain authority bonus
            if any(auth in domain.lower() for auth in ['.edu', 'harvard', 'university']):
                priority_score += 3
            elif any(news in domain.lower() for news in ['times', 'post', 'reuters', 'bloomberg', 'wsj']):
                priority_score += 2
            elif any(news in domain.lower() for news in ['cnn', 'bbc', 'npr', 'guardian']):
                priority_score += 1
            
            # Recency bonus (assuming Date column exists)
            date_str = row.get('Date', '')
            if '2025-06' in date_str:
                priority_score += 2
            elif '2025' in date_str:
                priority_score += 1
            
            articles.append({
                'id': row.get('news_id', 'unverified'),
                'title': row.get('title', 'No title'),
                'domain': domain,
                'date': date_str,
                'priority': priority_score
            })
        
        # Sort by priority score, then by date
        articles.sort(key=lambda x: (x['priority'], x['date']), reverse=True)
        return articles

    def get_strategy_performance(self) -> Dict[str, int]:
        """Get performance statistics for different strategies"""
        return self.strategy_history.copy()

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]