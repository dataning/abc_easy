import pandas as pd
import numpy as np
import os
import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch

# Core ML libraries - SentenceTransformer only (no M2V)
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# Force CPU fallback for MPS device issues
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

if torch.backends.mps.is_available():
    print("⚠️ Forcing CPU fallback for MPS-incompatible operations")
    torch.set_default_device('cpu')

# spaCy and NetworkX for enhanced keyword extraction
try:
    import spacy
    import networkx as nx
    ENHANCED_EXTRACTION_AVAILABLE = True
    print("✅ spaCy and NetworkX available for enhanced keyword extraction")
except ImportError:
    ENHANCED_EXTRACTION_AVAILABLE = False
    print("⚠️ spaCy/NetworkX not available - using fallback keyword extraction")

# Enhanced keyword extraction import
try:
    from enhanced_keyword_extractor import EnhancedKeywordExtractor
    ENHANCED_EXTRACTION_IMPORT = True
    print("✅ Enhanced keyword extraction module available")
except ImportError as e:
    ENHANCED_EXTRACTION_IMPORT = False
    print(f"⚠️ Enhanced extraction module not available: {e}")

# Project imports
from skywalker import get_and_save_gdelt_articles
from citation_tracker import CitationTracker

# Import all LLM providers
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Configuration - choose your provider
PROVIDER = "cerebras"  # Options: "openai", "cerebras", "gemini", "groq"

# API Keys and Models
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY"),
    "cerebras": "csk-mfjf25kf5v83hr4ercffhnx388mjr2rjy68kxxvpyn4e9nrt",
    "gemini": "AIzaSyC_kJEjiBGbGqOMshzsLC5DyDtutSNoZpQ",
    "groq": "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"
}

MODELS = {
    "openai": "gpt-4",
    "cerebras": "llama-4-scout-17b-16e-instruct",
    "gemini": "gemini-2.0-flash",
    "groq": "meta-llama/llama-4-maverick-17b-128e-instruct"
}

# Initialize client based on provider and availability
client = None
if PROVIDER == "openai" and OPENAI_AVAILABLE:
    client = OpenAI(api_key=API_KEYS["openai"])
elif PROVIDER == "cerebras" and CEREBRAS_AVAILABLE:
    client = Cerebras(api_key=API_KEYS["cerebras"])
elif PROVIDER == "gemini" and GEMINI_AVAILABLE:
    client = genai.Client(api_key=API_KEYS["gemini"])
elif PROVIDER == "groq" and GROQ_AVAILABLE:
    client = Groq(api_key=API_KEYS["groq"])
else:
    raise ValueError(f"Provider '{PROVIDER}' not available or not installed")

LLM_MODEL = MODELS[PROVIDER]

# Initialize SentenceTransformer embedding model (CPU-safe)
print("🔧 Initializing SentenceTransformer embedding model...")
try:
    # Force CPU to avoid MPS issues completely
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = embedder.to('cpu')  # Force CPU execution
    print("✅ SentenceTransformer all-MiniLM-L6-v2 loaded successfully on CPU")
except Exception as e:
    print(f"❌ Failed to load SentenceTransformer: {e}")
    raise e

# ========================= TOOL BASE CLASS =========================

class Tool(ABC):
    """Base class for tools agents can use"""
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        pass

# ========================= LLM TOOL =========================

class LLMTool(Tool):
    """Unified tool for LLM queries supporting multiple providers"""
    def name(self) -> str:
        return "llm_query"

    def description(self) -> str:
        return f"Query LLM for analysis, extraction, or generation (using {PROVIDER})"

    def execute(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        try:
            if PROVIDER == "openai":
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            
            elif PROVIDER == "cerebras":
                response = client.chat.completions.create(
                    messages=[{"role": "system", "content": prompt}],
                    model=LLM_MODEL,
                    stream=False,
                    max_completion_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            
            elif PROVIDER == "gemini":
                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt
                )
                return response.text.strip()
            
            elif PROVIDER == "groq":
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    stream=False
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"⚠️ {PROVIDER.title()} API error: {e}")
            return f"Error generating response: {str(e)}"

# ========================= NEWS SEARCH TOOL =========================

class NewsSearchTool(Tool):
    """Enhanced tool for searching news via GDELT with advanced keyword extraction"""
    
    def __init__(self):
        self.keyword_extractor = None
        self._last_extraction = {}
        
    def initialize_extractor(self, llm_tool):
        """Initialize the enhanced keyword extractor when LLM tool is available"""
        if ENHANCED_EXTRACTION_IMPORT:
            try:
                self.keyword_extractor = EnhancedKeywordExtractor(llm_tool)
                print("✅ Enhanced keyword extractor initialized")
                return True
            except Exception as e:
                print(f"⚠️ Could not initialize enhanced extractor: {e}")
                return False
        return False

    def name(self) -> str:
        return "news_search"

    def description(self) -> str:
        return "Search for news articles using advanced NLP keyword extraction"

    def execute(self, keywords: List[str] = None, llm_tool=None, max_records: int = 250, citation_tracker: CitationTracker = None) -> pd.DataFrame:
        # Initialize extractor if LLM tool is provided and extractor isn't ready
        if llm_tool and not self.keyword_extractor:
            self.initialize_extractor(llm_tool)
        
        # Enhanced extraction for simple keyword lists
        if (keywords and len(keywords) <= 3 and self.keyword_extractor and 
            all(len(kw) < 20 for kw in keywords)):
            
            question = f"How is {' '.join(keywords)}?"
            print(f"🧠 Using enhanced extraction for: {question}")
            
            try:
                extraction_result = self.keyword_extractor.extract_enhanced_keywords(question, max_keywords=8)
                search_keywords = extraction_result["search_variants"][:6]
                
                print(f"   🎯 Enhanced keywords: {search_keywords}")
                print(f"   📊 Extraction confidence: {extraction_result['confidence']:.1%}")
                
                # Store extraction results for debugging
                self._last_extraction = extraction_result
                
            except Exception as e:
                print(f"   ⚠️ Enhanced extraction failed: {e}")
                search_keywords = keywords
        else:
            search_keywords = keywords if isinstance(keywords, list) else [keywords]

        print(f"📋 Final search keywords: {search_keywords[:3]}...")
        
        # Execute searches
        all_articles = []
        search_stats = {"total_attempted": 0, "successful": 0, "failed": 0}
        
        for kw in search_keywords[:5]:
            search_stats["total_attempted"] += 1
            try:
                print(f"   🔍 Searching for: {kw}")
                df = get_and_save_gdelt_articles(query=kw, max_records=max_records//len(search_keywords[:5]))
                
                if df is not None and not df.empty:
                    processed_df = self._process_search_results(df, kw, citation_tracker)
                    if not processed_df.empty:
                        all_articles.append(processed_df)
                        search_stats["successful"] += 1
                        print(f"   ✅ Retrieved {len(processed_df)} articles for '{kw}'")
                    else:
                        search_stats["failed"] += 1
                        print(f"   ⚠️ No valid articles after processing for '{kw}'")
                else:
                    search_stats["failed"] += 1
                    print(f"   ⚠️ No articles found for '{kw}'")
                    
            except Exception as e:
                search_stats["failed"] += 1
                print(f"   ❌ Error searching for '{kw}': {e}")
                continue

        # Combine and deduplicate results (keep in memory only)
        if all_articles:
            combined = pd.concat(all_articles, ignore_index=True)
            if "url" in combined.columns:
                initial_count = len(combined)
                combined = combined.drop_duplicates(subset="url").reset_index(drop=True)
                final_count = len(combined)
                if initial_count > final_count:
                    print(f"   🔄 Removed {initial_count - final_count} duplicate articles")
            
            print(f"✅ Search completed: {search_stats['successful']}/{search_stats['total_attempted']} successful searches")
            print(f"📊 Final dataset: {len(combined)} unique articles (memory only)")
            return combined
        
        # Return empty DataFrame with proper columns
        base_columns = ["title", "domain", "Date", "url", "source_keyword"]
        if citation_tracker:
            base_columns.append("news_id")
        
        print(f"⚠️ No articles retrieved from any search terms")
        return pd.DataFrame(columns=base_columns)

    def _process_search_results(self, df: pd.DataFrame, keyword: str, citation_tracker: Optional[CitationTracker]) -> pd.DataFrame:
        """Process and clean search results"""
        if df.empty:
            return pd.DataFrame()
            
        required_columns = {"sourcecountry", "language", "Date", "title", "domain", "url"}
        
        if required_columns.issubset(set(df.columns)):
            # Apply filters for quality
            filtered = df[
                (df["sourcecountry"] == "United States") & 
                (df["language"] == "English") &
                (df["title"].str.len() > 10) &
                (df["domain"].notna())
            ].copy()
            
            if not filtered.empty:
                # Clean and standardize date format
                filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
                filtered = filtered.dropna(subset=["Date"])
                filtered["Date"] = filtered["Date"].dt.strftime("%Y-%m-%d")
                filtered["source_keyword"] = keyword
                
                # Add citation tracking
                if citation_tracker:
                    news_ids = []
                    for _, row in filtered.iterrows():
                        news_id = citation_tracker.add_article(
                            title=row.get("title", "No title"),
                            domain=row.get("domain", "Unknown"),
                            date=row.get("Date", "Unknown"),
                            url=row.get("url", "")
                        )
                        news_ids.append(news_id)
                    filtered["news_id"] = news_ids
                
                # Select relevant columns
                keep_columns = ["title", "domain", "Date", "url", "source_keyword"]
                if citation_tracker:
                    keep_columns.append("news_id")
                
                return filtered[keep_columns]
        else:
            # Handle DataFrames with missing required columns
            df_copy = df.copy()
            df_copy["source_keyword"] = keyword
            available_columns = [c for c in ["title", "domain", "Date", "url"] if c in df_copy.columns]
            
            if "title" in df_copy.columns and "url" in df_copy.columns and citation_tracker:
                news_ids = []
                for _, row in df_copy.iterrows():
                    news_id = citation_tracker.add_article(
                        title=row.get("title", "No title"),
                        domain=row.get("domain", "Unknown"),
                        date=row.get("Date", "Unknown"),
                        url=row.get("url", "")
                    )
                    news_ids.append(news_id)
                df_copy["news_id"] = news_ids
                available_columns.append("news_id")
            
            available_columns.append("source_keyword")
            return df_copy[available_columns] if available_columns else pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_last_extraction_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of the last keyword extraction"""
        return self._last_extraction

    def debug_last_extraction(self) -> None:
        """Debug the last keyword extraction process"""
        if self._last_extraction:
            extraction = self._last_extraction
            print(f"\n🔬 **LAST EXTRACTION DEBUG:**")
            print(f"   • Primary keywords: {extraction.get('primary_keywords', [])}")
            print(f"   • Search variants: {extraction.get('search_variants', [])}")
            print(f"   • Confidence: {extraction.get('confidence', 0):.1%}")
            print(f"   • Entities found: {len(extraction.get('entities', []))}")
            
            graph_analysis = extraction.get('graph_analysis', {})
            print(f"   • Graph nodes: {graph_analysis.get('nodes', 0)}")
            print(f"   • Graph edges: {graph_analysis.get('edges', 0)}")
            print(f"   • Graph density: {graph_analysis.get('density', 0):.3f}")
            
            llm_insights = extraction.get('llm_insights', {})
            print(f"   • LLM domain: {llm_insights.get('domain', 'unknown')}")
            print(f"   • LLM main subject: {llm_insights.get('main_subject', 'none')}")
        else:
            print(f"   ⚠️ No extraction data available")

# ========================= SEMANTIC FILTER TOOL =========================

class SemanticFilterTool(Tool):
    """Enhanced tool for semantic filtering using SentenceTransformer embeddings"""
    def name(self) -> str:
        return "semantic_filter"

    def description(self) -> str:
        return "Filter articles by semantic similarity using SentenceTransformer embeddings"

    def execute(self, articles_df: pd.DataFrame, query: str, top_k: int = 100, threshold: float = 0.3) -> pd.DataFrame:
        if articles_df.empty:
            print("   ⚠️ No articles to filter")
            return articles_df
        
        print(f"   🔍 Semantic filtering {len(articles_df)} articles using SentenceTransformer...")
        
        # Prepare text for embedding
        texts = []
        for _, row in articles_df.iterrows():
            title = row.get("title", "")
            domain = row.get("domain", "")
            text = f"{title} {domain}".strip()
            texts.append(text)
        
        try:
            # Generate embeddings using SentenceTransformer on CPU
            print(f"   🧠 Generating embeddings for {len(texts)} articles...")
            
            # Force CPU execution for embeddings
            with torch.device('cpu'):
                embeddings = embedder.encode(
                    texts, 
                    convert_to_numpy=True, 
                    show_progress_bar=False,
                    device='cpu'
                )
                
                # Generate query embedding
                print(f"   🎯 Generating query embedding...")
                query_embedding = embedder.encode(
                    [query], 
                    convert_to_numpy=True,
                    device='cpu'
                )
            
            # Ensure embeddings are float32 and normalized
            embeddings = embeddings.astype("float32")
            query_embedding = query_embedding.astype("float32")
            
            faiss.normalize_L2(embeddings)
            faiss.normalize_L2(query_embedding)
            
            # Create FAISS index
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            # Search for similar articles
            k = min(top_k, len(embeddings))
            similarities, indices = index.search(query_embedding, k)
            
            # Filter by similarity threshold
            valid_indices = []
            valid_similarities = []
            for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
                if sim >= threshold:
                    valid_indices.append(idx)
                    valid_similarities.append(sim)
            
            if valid_indices:
                filtered_df = articles_df.iloc[valid_indices].copy()
                filtered_df["similarity_score"] = valid_similarities
                filtered_df = filtered_df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
                
                print(f"   ✅ Filtered to {len(filtered_df)} articles (similarity >= {threshold:.2f})")
                print(f"   📊 Similarity range: {min(valid_similarities):.3f} - {max(valid_similarities):.3f}")
                
                return filtered_df
            else:
                print(f"   ⚠️ No articles met similarity threshold {threshold:.2f}")
                # Return top articles as fallback
                k_fallback = min(20, len(embeddings))
                _, fallback_indices = index.search(query_embedding, k_fallback)
                fallback_df = articles_df.iloc[fallback_indices[0]].copy()
                fallback_df["similarity_score"] = similarities[0][:k_fallback]
                print(f"   📉 Returning top {len(fallback_df)} articles as fallback")
                return fallback_df.reset_index(drop=True)
                
        except Exception as e:
            print(f"   ⚠️ Semantic filtering error: {e}")
            return articles_df.head(top_k).reset_index(drop=True)

# ========================= RELEVANCE EVALUATOR TOOL =========================

class RelevanceEvaluatorTool(Tool):
    """Enhanced tool for evaluating relevance with SentenceTransformer embeddings and multiple scoring methods"""
    def name(self) -> str:
        return "evaluate_relevance"

    def description(self) -> str:
        return "Evaluate relevance using SentenceTransformer embeddings, BM25, and semantic analysis"

    def execute(self, query: str, articles_df: pd.DataFrame, keywords: List[str]) -> Dict[str, Any]:
        if articles_df.empty:
            return {
                "score": 0.0, 
                "relevant_count": 0, 
                "high_relevance_count": 0,
                "total": 0,
                "confidence": 0.0,
                "method": "empty_dataset"
            }

        print(f"   📊 Evaluating relevance for {len(articles_df)} articles using SentenceTransformer...")

        try:
            # Prepare text data
            titles = articles_df['title'].tolist()
            domains = articles_df.get('domain', pd.Series([''] * len(articles_df))).tolist()
            
            # Enhanced semantic scoring using SentenceTransformer embeddings
            semantic_scores = self._calculate_semantic_relevance_cpu_safe(query, titles, domains)
            
            # BM25 scoring
            tokenized_corpus = [title.lower().split() for title in titles]
            tokenized_query = " ".join(keywords).lower().split() if keywords else query.lower().split()
            
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Normalize BM25 scores
            if len(bm25_scores) > 0 and np.max(bm25_scores) > 0:
                bm25_scores = bm25_scores / np.max(bm25_scores)
            else:
                bm25_scores = np.zeros(len(titles))
            
            # Domain relevance scoring
            domain_scores = self._calculate_domain_relevance(domains, query, keywords)
            
            # Combine scores with enhanced weighting for semantic similarity
            combined_scores = (0.5 * np.array(semantic_scores) + 
                             0.3 * bm25_scores + 
                             0.2 * np.array(domain_scores))
            
            # Calculate statistics
            if len(combined_scores) > 0:
                median_score = np.median(combined_scores)
                mean_score = np.mean(combined_scores)
                std_score = np.std(combined_scores)
                
                # Adaptive thresholding
                threshold_moderate = max(median_score, mean_score - 0.5 * std_score)
                threshold_high = max(median_score * 1.5, mean_score + 0.5 * std_score)
                
                relevant_count = sum(1 for s in combined_scores if s > threshold_moderate)
                high_relevance_count = sum(1 for s in combined_scores if s > threshold_high)
                
                # Enhanced confidence calculation
                base_confidence = min(relevant_count / len(articles_df), 1.0) if len(articles_df) > 0 else 0
                
                # Boost confidence based on semantic similarity quality
                semantic_boost = min(np.mean(semantic_scores) * 0.3, 0.3)
                confidence = min(base_confidence + semantic_boost, 1.0)
                
                # Additional boost if we have high-relevance articles
                if high_relevance_count > 0:
                    confidence = min(confidence * 1.2, 1.0)
                
                result = {
                    "score": float(mean_score),
                    "relevant_count": int(relevant_count),
                    "high_relevance_count": int(high_relevance_count),
                    "total": len(articles_df),
                    "confidence": float(confidence),
                    "method": "sentencetransformer_bm25_domain_combined",
                    "statistics": {
                        "median": float(median_score),
                        "std": float(std_score),
                        "max": float(np.max(combined_scores)),
                        "min": float(np.min(combined_scores)),
                        "semantic_avg": float(np.mean(semantic_scores)),
                        "bm25_avg": float(np.mean(bm25_scores)),
                        "domain_avg": float(np.mean(domain_scores))
                    }
                }
                
                print(f"   ✅ SentenceTransformer Relevance: {confidence:.1%} confidence, {relevant_count}/{len(articles_df)} relevant")
                print(f"   📈 Semantic avg: {np.mean(semantic_scores):.3f}, BM25 avg: {np.mean(bm25_scores):.3f}")
                return result
            else:
                return {
                    "score": 0.0,
                    "relevant_count": 0,
                    "high_relevance_count": 0,
                    "total": len(articles_df),
                    "confidence": 0.0,
                    "method": "no_scores_generated"
                }
                
        except Exception as e:
            print(f"   ⚠️ SentenceTransformer relevance evaluation error: {e}")
            # Fallback to simple keyword matching
            return self._fallback_relevance_evaluation(query, articles_df, keywords)

    def _calculate_semantic_relevance_cpu_safe(self, query: str, titles: List[str], domains: List[str]) -> List[float]:
        """Calculate semantic relevance using SentenceTransformer embeddings with CPU-safe execution"""
        try:
            # Combine titles and domains for richer context
            texts = [f"{title} {domain}".strip() for title, domain in zip(titles, domains)]
            
            # Generate embeddings on CPU
            with torch.device('cpu'):
                text_embeddings = embedder.encode(
                    texts, 
                    convert_to_numpy=True,
                    device='cpu',
                    show_progress_bar=False
                ).astype("float32")
                
                query_embedding = embedder.encode(
                    [query], 
                    convert_to_numpy=True,
                    device='cpu',
                    show_progress_bar=False
                ).astype("float32")
            
            # Normalize embeddings
            faiss.normalize_L2(text_embeddings)
            faiss.normalize_L2(query_embedding)
            
            # Calculate cosine similarities
            similarities = np.dot(text_embeddings, query_embedding.T).flatten()
            
            # Convert to 0-1 scale for better interpretation
            normalized_similarities = (similarities + 1) / 2  # From [-1,1] to [0,1]
            
            return normalized_similarities.tolist()
            
        except Exception as e:
            print(f"   ⚠️ Semantic relevance calculation failed: {e}")
            # Return neutral scores if embeddings fail
            return [0.5] * len(titles)

    def _calculate_domain_relevance(self, domains: List[str], query: str, keywords: List[str]) -> List[float]:
        """Calculate relevance scores based on domain authority and topical relevance"""
        domain_scores = []
        
        # Domain authority mapping
        authority_domains = {
            'harvard.edu': 1.0, 'edu': 0.8, 'gov': 0.9,
            'nytimes.com': 0.9, 'wsj.com': 0.9, 'bloomberg.com': 0.9,
            'reuters.com': 0.8, 'bbc.com': 0.8, 'cnn.com': 0.7,
            'washingtonpost.com': 0.8, 'ft.com': 0.9, 'economist.com': 0.9
        }
        
        # Topical relevance keywords
        topic_keywords = set(k.lower() for k in (keywords or [])) | set(query.lower().split())
        
        for domain in domains:
            score = 0.5  # Base score
            domain_lower = domain.lower() if domain else ""
            
            # Authority bonus
            for auth_domain, bonus in authority_domains.items():
                if auth_domain in domain_lower:
                    score = max(score, bonus)
                    break
            
            # Topical relevance bonus
            for keyword in topic_keywords:
                if keyword in domain_lower:
                    score = min(score + 0.2, 1.0)
            
            domain_scores.append(score)
        
        return domain_scores

    def _fallback_relevance_evaluation(self, query: str, articles_df: pd.DataFrame, keywords: List[str]) -> Dict[str, Any]:
        """Simple fallback relevance evaluation using keyword matching"""
        titles = articles_df['title'].tolist()
        all_keywords = set(k.lower() for k in (keywords or [])) | set(query.lower().split())
        
        relevant_count = 0
        for title in titles:
            title_lower = title.lower()
            if any(keyword in title_lower for keyword in all_keywords):
                relevant_count += 1
        
        confidence = relevant_count / len(articles_df) if len(articles_df) > 0 else 0
        
        return {
            "score": confidence,
            "relevant_count": relevant_count,
            "high_relevance_count": max(1, relevant_count // 2),
            "total": len(articles_df),
            "confidence": confidence,
            "method": "keyword_fallback"
        }

# ========================= UTILITY FUNCTIONS =========================

def test_enhanced_extraction(question: str = "How is Harvard's endowment situation?"):
    """Test function to verify enhanced keyword extraction is working"""
    print(f"\n🧪 **TESTING ENHANCED KEYWORD EXTRACTION**")
    print(f"Testing with: '{question}'")
    print("=" * 60)
    
    # Create tools
    llm_tool = LLMTool()
    news_tool = NewsSearchTool()
    
    # Initialize the enhanced extractor
    success = news_tool.initialize_extractor(llm_tool)
    
    if success and news_tool.keyword_extractor:
        print(f"✅ Enhanced extractor initialized")
        
        try:
            # Test extraction
            result = news_tool.keyword_extractor.extract_enhanced_keywords(question)
            
            if result.get('primary_keywords'):
                print(f"\n🎉 **EXTRACTION SUCCESSFUL!**")
                print(f"   • Primary keywords: {result['primary_keywords']}")
                print(f"   • Search variants: {result['search_variants']}")
                print(f"   • Confidence: {result['confidence']:.1%}")
                return True
            else:
                print(f"\n❌ **EXTRACTION FAILED:** No keywords generated")
                return False
                
        except Exception as e:
            print(f"\n❌ **TEST FAILED:** {e}")
            return False
    else:
        print(f"❌ Enhanced extractor not available")
        return False

def test_semantic_operations():
    """Test semantic operations with SentenceTransformer"""
    print(f"\n🧪 **TESTING SEMANTIC OPERATIONS**")
    print("=" * 50)
    
    try:
        # Test basic embedding generation
        test_texts = ["Harvard University endowment", "Investment fund performance", "Educational finance"]
        
        print(f"   🔍 Testing embedding generation...")
        with torch.device('cpu'):
            embeddings = embedder.encode(test_texts, convert_to_numpy=True, device='cpu', show_progress_bar=False)
        
        print(f"   ✅ Generated embeddings: {embeddings.shape}")
        
        # Test similarity calculation
        query = "How is Harvard's financial situation?"
        with torch.device('cpu'):
            query_emb = embedder.encode([query], convert_to_numpy=True, device='cpu', show_progress_bar=False)
        
        similarities = np.dot(embeddings, query_emb.T).flatten()
        print(f"   ✅ Calculated similarities: {similarities}")
        
        print(f"\n🎉 **SEMANTIC OPERATIONS WORKING!**")
        return True
        
    except Exception as e:
        print(f"\n❌ **SEMANTIC OPERATIONS FAILED:** {e}")
        return False

def debug_extraction_system(question: str):
    """Debug the entire extraction system"""
    print(f"\n🔬 **DEBUGGING EXTRACTION SYSTEM**")
    print(f"Question: '{question}'")
    print("=" * 70)
    
    # Test semantic operations first
    semantic_working = test_semantic_operations()
    
    # Test enhanced extraction
    extraction_working = test_enhanced_extraction(question)
    
    # Test tools integration
    print(f"\n🔧 **TESTING TOOLS INTEGRATION**")
    try:
        llm_tool = LLMTool()
        news_tool = NewsSearchTool()
        relevance_tool = RelevanceEvaluatorTool()
        
        print(f"   ✅ All tools instantiated successfully")
        
        # Test a simple search
        simple_keywords = question.split()[:2]
        print(f"   🔍 Testing search with keywords: {simple_keywords}")
        
        # This would test the full pipeline
        result_df = news_tool.execute(keywords=simple_keywords, llm_tool=llm_tool, max_records=50)
        print(f"   📊 Search returned {len(result_df)} articles (stored in memory only)")
        
        return {
            "semantic_operations": semantic_working,
            "enhanced_extraction": extraction_working,
            "tools_integration": True,
            "search_results": len(result_df)
        }
        
    except Exception as e:
        print(f"   ❌ Tools integration failed: {e}")
        return {
            "semantic_operations": semantic_working,
            "enhanced_extraction": extraction_working,
            "tools_integration": False,
            "error": str(e)
        }

# ========================= EXPORT FOR AGENT USE =========================

if __name__ == "__main__":
    # Run tests when script is executed directly
    print("🧪 Running system tests...")
    test_semantic_operations()
    test_enhanced_extraction()
    print("✅ All tests completed!")