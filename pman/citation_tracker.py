import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

@dataclass
class Citation:
    """Represents a citation with verification status"""
    news_id: str
    title: str
    domain: str
    date: str
    url: str
    verified: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

class CitationTracker:
    """Enhanced citation tracker with verification and hallucination detection"""
    def __init__(self):
        self.id_to_meta: Dict[str, Citation] = {}
        self.url_to_id: Dict[str, str] = {}
        self.hallucination_count = 0
        
        # Add compatibility attributes for new interface
        self.citations = {}  # url -> citation info (for compatibility)
        self.citation_counter = 0
        self.used_citations = []
        self.hallucinations = []

    def generate_news_id(self, title: str, url: str) -> str:
        content = f"{title}||{url}"
        news_id = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"news_{news_id}"

    def add_article(self, title: str, domain: str, date: str, url: str) -> str:
        if url in self.url_to_id:
            return self.url_to_id[url]
        news_id = self.generate_news_id(title, url)
        citation = Citation(news_id=news_id, title=title, domain=domain, date=date, url=url)
        self.id_to_meta[news_id] = citation
        self.url_to_id[url] = news_id
        
        # Also update new interface for compatibility
        self.citation_counter += 1
        self.citations[url] = {
            'id': self.citation_counter,
            'title': title,
            'url': url,
            'date': date,
            'domain': domain,
            'used': False,
            'news_id': news_id  # Keep reference to original ID
        }
        
        return news_id

    def track_articles(self, articles_df):
        """Track articles and assign citation IDs - FIXED VERSION"""
        if articles_df is None or (hasattr(articles_df, 'empty') and articles_df.empty):
            return
            
        try:
            if hasattr(articles_df, 'iterrows'):  # DataFrame
                for idx, article in articles_df.iterrows():
                    url = article.get('url', '')
                    title = str(article.get('title', 'No title'))[:200]
                    domain = str(article.get('domain', 'Unknown'))[:50]
                    date = str(article.get('Date', 'Unknown'))[:10]
                    
                    if url and url not in self.url_to_id:  # Use existing check
                        # Use the existing add_article method
                        self.add_article(title, domain, date, url)
                        
        except Exception as e:
            print(f"   ⚠️ Citation tracking failed: {e}")

    def get_citation_id(self, url):
        """Get citation ID for a URL - FIXED VERSION"""
        # Try new interface first
        if url in self.citations:
            return self.citations[url]['id']
        # Fallback to old interface
        if url in self.url_to_id:
            news_id = self.url_to_id[url]
            # Extract numeric part from news_id for compatibility
            if news_id in self.id_to_meta:
                # Use the citation counter as the display ID
                return self.citations.get(url, {}).get('id', 1)
        return None

    def mark_as_used(self, url):
        """Mark citation as used in answer"""
        if url in self.citations:
            self.citations[url]['used'] = True
            if self.citations[url] not in self.used_citations:
                self.used_citations.append(self.citations[url])

    def get_used_citations(self, answer_text: str = None):
        """Get list of used citations for display - UNIFIED VERSION"""
        try:
            # If answer_text provided, use old interface logic
            if answer_text is not None:
                return self._get_used_citations_from_text(answer_text)
            
            # Otherwise use new interface - return citations marked as used
            used = [cit for cit in self.citations.values() if cit.get('used', False)]
            if used:
                return used
            
            # Fallback: return recent citations from old interface
            return self.get_most_recent_citations(limit=10)
            
        except Exception as e:
            print(f"   ⚠️ Error getting used citations: {e}")
            return []

    def _get_used_citations_from_text(self, answer_text: str) -> List[Dict[str, str]]:
        """Extract citations from answer text (old interface)"""
        pattern = r'\[news_[a-f0-9]{8}\]'
        found = re.findall(pattern, answer_text)
        unique_ids = list({f[1:-1] for f in found})
        used = []
        for nid in unique_ids:
            if nid in self.id_to_meta:
                c = self.id_to_meta[nid]
                used.append({
                    "id": nid,
                    "title": c.title,
                    "domain": c.domain,
                    "date": c.date,
                    "url": c.url
                })
        used.sort(key=lambda x: x["date"], reverse=True)
        return used

    def get_citation_list(self):
        """Get all citations as a list - FIXED VERSION"""
        try:
            # Try new interface first
            if self.citations:
                return list(self.citations.values())
            
            # Fallback to old interface
            out = []
            for nid, cite in self.id_to_meta.items():
                out.append({
                    "id": nid,
                    "title": cite.title,
                    "domain": cite.domain,
                    "date": cite.date,
                    "url": cite.url
                })
            return out
            
        except Exception as e:
            print(f"   ⚠️ Error getting citation list: {e}")
            return []

    def validate_citations_in_text(self, text, available_urls):
        """Validate citations with proper error handling - FIXED VERSION"""
        import re
        
        try:
            # Find citation patterns like [1], [2], etc. (new format)
            new_citation_refs = re.findall(r'\[(\d+)\]', str(text))
            # Find citation patterns like [news_xxxxxxxx] (old format)
            old_citation_refs = re.findall(r'\[news_[a-f0-9]{8}\]', str(text))
            
            valid_citations = []
            hallucinations = []
            
            # Handle new format citations [1], [2], etc.
            for ref in new_citation_refs:
                try:
                    citation_id = int(ref)
                    # Find citation by ID in new interface
                    found = False
                    for url, citation_info in self.citations.items():
                        if citation_info.get('id') == citation_id and url in available_urls:
                            valid_citations.append(citation_info)
                            self.mark_as_used(url)
                            found = True
                            break
                    
                    if not found:
                        hallucinations.append(f"[{ref}]")
                        
                except (ValueError, TypeError):
                    continue
            
            # Handle old format citations [news_xxxxxxxx]
            for ref in old_citation_refs:
                news_id = ref[1:-1]  # Remove brackets
                if news_id in self.id_to_meta:
                    cite = self.id_to_meta[news_id]
                    if cite.url in available_urls:
                        valid_citations.append({
                            'id': news_id,
                            'title': cite.title,
                            'domain': cite.domain,
                            'date': cite.date,
                            'url': cite.url,
                            'used': True
                        })
                        # Also mark in new interface
                        if cite.url in self.citations:
                            self.mark_as_used(cite.url)
                else:
                    hallucinations.append(ref)
            
            self.hallucinations.extend(hallucinations)
            return valid_citations, hallucinations
            
        except Exception as e:
            print(f"   ⚠️ Citation validation failed: {e}")
            return [], []

    def verify_citations(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Basic citation verification (legacy method)"""
        pattern = r'\[news_[a-f0-9]{8}\]'
        found = re.findall(pattern, text)
        verified_text = text
        stats = {
            "total_citations": len(found),
            "valid_citations": 0,
            "invalid_citations": 0,
            "hallucinations": []
        }
        for ref in found:
            nid = ref[1:-1]
            if nid in self.id_to_meta:
                stats["valid_citations"] += 1
            else:
                verified_text = verified_text.replace(ref, "[unverified]")
                stats["invalid_citations"] += 1
                stats["hallucinations"].append(nid)
                self.hallucination_count += 1
        return verified_text, stats

    def get_domain_diversity(self) -> Dict[str, int]:
        """Get distribution of articles by domain"""
        domain_counts = {}
        for citation in self.id_to_meta.values():
            domain_counts[citation.domain] = domain_counts.get(citation.domain, 0) + 1
        return domain_counts

    def get_most_recent_citations(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get most recent citations for priority use"""
        citations = list(self.id_to_meta.values())
        citations.sort(key=lambda x: x.date, reverse=True)
        return [{
            "id": c.news_id,
            "title": c.title,
            "domain": c.domain,
            "date": c.date,
            "url": c.url
        } for c in citations[:limit]]

    # ENHANCED CITATION VERIFICATION METHODS
    def verify_citations_against_content(self, answer_text, articles_df=None):
        """ENHANCED: Verify that citations actually support the claims made in the answer"""
        verification_results = {
            'verified_citations': [],
            'hallucinated_citations': [],
            'unsupported_claims': [],
            'verification_score': 0.0,
            'detailed_analysis': []
        }
        
        try:
            # Import here to avoid dependency issues
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer('all-MiniLM-L6-v2')
            semantic_available = True
            print("   🧠 Using semantic verification with sentence transformers")
        except Exception as e:
            print(f"   ⚠️ Semantic verification not available: {e}")
            semantic_available = False
        
        # Extract citations from answer
        cited_ids = self._extract_citations_from_text(answer_text)
        if not cited_ids:
            print("   ℹ️ No citations found in answer text")
            return verification_results
        
        print(f"   🔍 Verifying {len(cited_ids)} citations...")
        
        # Split answer into sentences and find citation references
        sentences = re.split(r'[.!?]+', answer_text)
        verification_count = 0
        supported_count = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Find citations in this sentence
            sentence_citations = self._find_citations_in_sentence(sentence)
            if not sentence_citations:
                continue
                
            verification_count += 1
            sentence_supported = False
            
            for citation_id in sentence_citations:
                # Get citation data
                citation_data = self._get_citation_data(citation_id, articles_df)
                
                if citation_data:
                    # Extract claim from sentence (remove citation references)
                    claim = re.sub(r'\[news_[a-f0-9]+\]|\[[a-f0-9]+\]|\[\d+\]', '', sentence).strip()
                    
                    # Get article content
                    article_title = citation_data.get('title', '')
                    article_content = f"{article_title}"
                    
                    # Verify claim against content
                    if semantic_available:
                        verification_score, similarity, keyword_overlap = self._semantic_verification(
                            claim, article_content, model
                        )
                        method = 'semantic'
                    else:
                        verification_score = self._keyword_verification(claim, article_content)
                        similarity = 0.0
                        keyword_overlap = verification_score
                        method = 'keyword'
                    
                    analysis_item = {
                        'citation_id': citation_id,
                        'claim': claim[:100] + '...' if len(claim) > 100 else claim,
                        'article_title': article_title,
                        'verification_score': float(verification_score),
                        'semantic_similarity': float(similarity),
                        'keyword_overlap': float(keyword_overlap),
                        'supported': verification_score > 0.3,
                        'method': method
                    }
                    
                    verification_results['detailed_analysis'].append(analysis_item)
                    
                    if verification_score > 0.3:  # Threshold for support
                        verification_results['verified_citations'].append(citation_id)
                        sentence_supported = True
                    else:
                        verification_results['hallucinated_citations'].append({
                            'citation_id': citation_id,
                            'claim': claim,
                            'reason': f'Low relevance (score: {verification_score:.2f})',
                            'article_title': article_title
                        })
                else:
                    verification_results['hallucinated_citations'].append({
                        'citation_id': citation_id,
                        'claim': re.sub(r'\[.*?\]', '', sentence).strip(),
                        'reason': 'Citation not found in source data'
                    })
            
            if sentence_supported:
                supported_count += 1
            else:
                verification_results['unsupported_claims'].append(sentence.strip())
        
        # Calculate overall verification score
        if verification_count > 0:
            verification_results['verification_score'] = supported_count / verification_count
        
        print(f"   📊 Verification complete: {supported_count}/{verification_count} sentences supported")
        print(f"   🎯 Verification rate: {verification_results['verification_score']:.1%}")
        
        return verification_results

    def _extract_citations_from_text(self, text):
        """Extract all citation IDs from text"""
        citation_patterns = [
            r'\[news_([a-f0-9]+)\]',  # [news_abc123]
            r'\[([a-f0-9]{6,})\]',    # [abc123]
            r'\[(\d+)\]'              # [123456] (numeric only)
        ]
        
        citations = set()
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.update(matches)
        
        return list(citations)

    def _find_citations_in_sentence(self, sentence):
        """Find citations within a specific sentence"""
        citation_patterns = [
            r'\[news_([a-f0-9]+)\]',
            r'\[([a-f0-9]{6,})\]',
            r'\[(\d+)\]'
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        
        return citations

    def _get_citation_data(self, citation_id, articles_df=None):
        """Get citation data from multiple sources"""
        # Try new interface first
        for url, citation_info in self.citations.items():
            if str(citation_info.get('id')) == citation_id:
                return citation_info
        
        # Try old interface
        clean_id = citation_id.replace('news_', '')
        if f"news_{clean_id}" in self.id_to_meta:
            citation = self.id_to_meta[f"news_{clean_id}"]
            return {
                'id': citation_id,
                'title': citation.title,
                'domain': citation.domain,
                'date': citation.date,
                'url': citation.url
            }
        
        # Try to find in articles_df
        if articles_df is not None and not articles_df.empty:
            return self._find_in_dataframe(citation_id, articles_df)
        
        return None

    def _find_in_dataframe(self, citation_id, articles_df):
        """Find citation data in the articles DataFrame"""
        for _, row in articles_df.iterrows():
            row_str = ' '.join(str(val) for val in row.values)
            if citation_id in row_str or citation_id[:6] in row_str:
                return {
                    'id': citation_id,
                    'title': row.get('title', 'No title'),
                    'domain': row.get('domain', 'Unknown'),
                    'date': row.get('Date', 'Unknown'),
                    'url': row.get('url', '#')
                }
        return None

    def _semantic_verification(self, claim, article_content, model):
        """Verify claim using semantic similarity"""
        try:
            import numpy as np
            claim_embedding = model.encode([claim])
            content_embedding = model.encode([article_content])
            similarity = float(np.dot(claim_embedding, content_embedding.T)[0][0])
            
            # Also check for keyword overlap
            claim_words = set(claim.lower().split())
            content_words = set(article_content.lower().split())
            keyword_overlap = len(claim_words.intersection(content_words)) / max(len(claim_words), 1)
            
            # Combined score (70% semantic, 30% keyword)
            combined_score = (similarity * 0.7) + (keyword_overlap * 0.3)
            return combined_score, similarity, keyword_overlap
            
        except Exception as e:
            print(f"   ⚠️ Semantic verification failed: {e}")
            keyword_score = self._keyword_verification(claim, article_content)
            return keyword_score, 0.0, keyword_score

    def _keyword_verification(self, claim, article_content):
        """Fallback keyword-based verification"""
        claim_words = set(claim.lower().split())
        content_words = set(article_content.lower().split())
        
        # Basic keyword overlap score
        if len(claim_words) == 0:
            return 0.0
        
        overlap = len(claim_words.intersection(content_words))
        return overlap / len(claim_words)

    def calculate_enhanced_quality_score(self, original_results, verification_results):
        """Calculate enhanced quality score including citation verification"""
        
        # Original quality components
        eval_stats = original_results.get('evaluation_stats', {})
        original_score = eval_stats.get('final_score', 0.0)
        
        # Citation verification components
        citation_accuracy = verification_results.get('verification_score', 0.0)
        hallucination_count = len(verification_results.get('hallucinated_citations', []))
        hallucination_penalty = min(hallucination_count * 0.15, 0.6)  # Cap at 0.6
        
        # Content relevance penalty
        answer_text = original_results.get('answer', '')
        relevance_penalty = 0.0
        
        # Penalize answers that admit irrelevance
        irrelevance_phrases = [
            'not discussed in any of the provided news articles',
            'no information available',
            'limited information available',
            'cannot provide specific',
            'would require access to'
        ]
        
        if any(phrase in answer_text.lower() for phrase in irrelevance_phrases):
            relevance_penalty = 0.4
        
        # Calculate enhanced score
        enhanced_score = (
            original_score * 0.3 +           # Original LLM evaluation (30%)
            citation_accuracy * 0.5 +        # Citation accuracy (50%) - MAJOR WEIGHT
            0.2                              # Base score (20%)
            - hallucination_penalty          # Hallucination penalty
            - relevance_penalty              # Relevance penalty
        )
        
        # Ensure score is between 0 and 1
        enhanced_score = max(0.0, min(1.0, enhanced_score))
        
        # Determine new verdict with stricter thresholds
        if enhanced_score >= 0.85:
            verdict = "Excellent"
        elif enhanced_score >= 0.65:
            verdict = "Good"
        elif enhanced_score >= 0.4:
            verdict = "Fair"
        else:
            verdict = "Poor"
        
        return {
            'enhanced_score': enhanced_score,
            'enhanced_verdict': verdict,
            'original_score': original_score,
            'citation_accuracy': citation_accuracy,
            'hallucination_penalty': hallucination_penalty,
            'relevance_penalty': relevance_penalty,
            'hallucination_count': hallucination_count,
            'scoring_breakdown': {
                'content_quality': original_score * 0.3,
                'citation_accuracy': citation_accuracy * 0.5,
                'base_score': 0.2,
                'hallucination_deduction': -hallucination_penalty,
                'relevance_deduction': -relevance_penalty
            }
        }

    # Compatibility method for app.py
    def track_citation(self, url, title, domain, date, citation_id=None):
        """Compatibility method - redirects to add_article"""
        return self.add_article(title=title, domain=domain, date=date, url=url)