# enhanced_keyword_extractor.py - Advanced keyword extraction using LLM + Graph approach

import re
import json
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

# Fallback imports - work without spaCy if needed
try:
    import spacy
    import networkx as nx
    ADVANCED_NLP_AVAILABLE = True
    print("✅ spaCy and NetworkX available for enhanced extraction")
except ImportError as e:
    ADVANCED_NLP_AVAILABLE = False
    print(f"⚠️ Advanced NLP not available: {e}")

@dataclass
class Entity:
    """Represents an extracted entity with its properties"""
    text: str
    label: str  # PERSON, ORG, GPE, etc.
    start: int
    end: int
    confidence: float = 1.0
    
@dataclass
class KeywordNode:
    """Represents a keyword node in the graph"""
    text: str
    entity_type: str
    importance: float
    aliases: Set[str]
    context: str

class EnhancedKeywordExtractor:
    """Advanced keyword extraction using LLM reasoning + optional spaCy + Graph structures"""
    
    def __init__(self, llm_tool):
        self.llm_tool = llm_tool
        self.nlp = None
        
        # Try to load spaCy model
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ Loaded spaCy en_core_web_sm model")
                self.advanced_mode = True
            except OSError:
                print("⚠️ spaCy model not found - using LLM-only mode")
                self.advanced_mode = False
        else:
            self.advanced_mode = False
        
        # Initialize knowledge graph
        if ADVANCED_NLP_AVAILABLE:
            self.keyword_graph = nx.Graph()
        else:
            self.keyword_graph = None
        
        # Domain-specific entity mappings
        self.domain_entities = {
            "education": ["university", "college", "endowment", "tuition", "academic", "harvard", "yale", "mit"],
            "finance": ["investment", "fund", "market", "economy", "trading", "billion", "million"],
            "politics": ["government", "policy", "election", "administration", "congress", "trump", "biden"],
            "technology": ["artificial intelligence", "AI", "software", "innovation", "startup", "tech"],
            "health": ["healthcare", "medical", "pharmaceutical", "hospital", "research", "drug"]
        }

    def extract_enhanced_keywords(self, question: str, max_keywords: int = 8) -> Dict[str, Any]:
        """Main method to extract keywords using the enhanced approach"""
        print(f"🔍 Enhanced keyword extraction from: '{question}'")
        
        if self.advanced_mode and self.nlp:
            # Step 1: spaCy NLP Analysis
            nlp_analysis = self._analyze_with_spacy(question)
            
            # Step 2: LLM Semantic Understanding
            llm_analysis = self._analyze_with_llm(question)
            
            # Step 3: Build Keyword Graph
            keyword_graph = self._build_keyword_graph(question, nlp_analysis, llm_analysis)
            
            # Step 4: Extract Final Keywords
            final_keywords = self._extract_final_keywords(keyword_graph, max_keywords)
            
            # Step 5: Generate Search Variants
            search_variants = self._generate_search_variants(final_keywords, question)
            
            result = {
                "primary_keywords": final_keywords,
                "search_variants": search_variants,
                "entities": nlp_analysis["entities"],
                "graph_analysis": self._analyze_graph(keyword_graph),
                "llm_insights": llm_analysis,
                "confidence": self._calculate_extraction_confidence(nlp_analysis, llm_analysis)
            }
        else:
            # Fallback to LLM-only mode
            result = self._llm_only_extraction(question, max_keywords)
        
        print(f"✅ Extracted {len(result['primary_keywords'])} primary keywords with {len(result['search_variants'])} variants")
        return result

    def _llm_only_extraction(self, question: str, max_keywords: int) -> Dict[str, Any]:
        """Fallback extraction using only LLM when spaCy is not available"""
        print("   🤖 Using LLM-only extraction mode")
        
        prompt = f"""Extract search keywords for this research question:

QUESTION: "{question}"

Analyze and extract:
1. PRIMARY KEYWORDS: 3-5 main entities/concepts for search
2. SEARCH VARIANTS: 6-10 alternative search terms and combinations
3. DOMAIN: The subject area (education, finance, politics, etc.)
4. ENTITIES: Important named entities (people, organizations, places)

Format as JSON:
{{
  "primary_keywords": ["keyword1", "keyword2", "keyword3"],
  "search_variants": ["variant1", "variant2", "variant3", "variant4", "variant5"],
  "domain": "domain_name",
  "entities": ["entity1", "entity2"],
  "confidence": 0.8
}}"""

        try:
            response = self.llm_tool.execute(prompt, max_tokens=300, temperature=0.2)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                return {
                    "primary_keywords": analysis.get("primary_keywords", [])[:max_keywords//2],
                    "search_variants": analysis.get("search_variants", [])[:max_keywords],
                    "entities": [{"text": e, "label": "ENTITY"} for e in analysis.get("entities", [])],
                    "graph_analysis": {"nodes": len(analysis.get("primary_keywords", [])), "edges": 0},
                    "llm_insights": analysis,
                    "confidence": analysis.get("confidence", 0.7)
                }
            else:
                print("   ⚠️ LLM response format error, using simple fallback")
                return self._simple_fallback_extraction(question, max_keywords)
                
        except Exception as e:
            print(f"   ⚠️ LLM extraction failed: {e}")
            return self._simple_fallback_extraction(question, max_keywords)

    def _simple_fallback_extraction(self, question: str, max_keywords: int) -> Dict[str, Any]:
        """Simple fallback when all else fails"""
        words = question.lower().replace('?', '').split()
        keywords = [w for w in words if len(w) > 3 and w not in ['what', 'how', 'when', 'where', 'why', 'does', 'will']][:max_keywords]
        
        return {
            "primary_keywords": keywords[:3],
            "search_variants": keywords + [f"{keywords[0]} news" if keywords else "news"],
            "entities": [],
            "graph_analysis": {"nodes": len(keywords), "edges": 0},
            "llm_insights": {"domain": "general"},
            "confidence": 0.5
        }

    def _analyze_with_spacy(self, text: str) -> Dict[str, Any]:
        """Analyze text using spaCy to extract entities, dependencies, and linguistic features"""
        if not self.nlp:
            return {"entities": [], "noun_phrases": [], "key_nouns": [], "proper_nouns": [], "dependencies": []}
            
        doc = self.nlp(text)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.0
            })
        
        # Extract noun phrases
        noun_phrases = [chunk.text.lower().strip() for chunk in doc.noun_chunks 
                       if len(chunk.text.strip()) > 2]
        
        # Extract key nouns and proper nouns
        key_nouns = []
        proper_nouns = []
        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop and len(token.text) > 2:
                key_nouns.append(token.lemma_.lower())
            elif token.pos_ == "PROPN" and not token.is_stop:
                proper_nouns.append(token.text)
        
        # Extract dependency relationships
        dependencies = []
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj", "compound"] and token.head.text != token.text:
                dependencies.append({
                    "dependent": token.text,
                    "head": token.head.text,
                    "relation": token.dep_
                })
        
        return {
            "entities": entities,
            "noun_phrases": list(set(noun_phrases)),
            "key_nouns": list(set(key_nouns)),
            "proper_nouns": list(set(proper_nouns)),
            "dependencies": dependencies
        }

    def _analyze_with_llm(self, question: str) -> Dict[str, Any]:
        """Use LLM to understand semantic context and extract conceptual keywords"""
        
        prompt = f"""Analyze this research question and extract key information for news search:

QUESTION: "{question}"

Please provide:
1. MAIN_SUBJECT: The primary entity/topic (person, organization, concept)
2. KEY_CONCEPTS: Important related concepts/themes (max 5)
3. SEARCH_FOCUS: What type of information is being sought
4. DOMAIN: The general domain (education, finance, politics, etc.)
5. RELATIONSHIPS: How the concepts relate to each other

Format as JSON:
{{
  "main_subject": "...",
  "key_concepts": ["...", "..."],
  "search_focus": "...",
  "domain": "...",
  "relationships": ["relationship1", "relationship2"]
}}"""

        try:
            response = self.llm_tool.execute(prompt, max_tokens=200, temperature=0.2)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                print(f"   🤖 LLM identified domain: {analysis.get('domain', 'unknown')}")
                print(f"   🎯 Main subject: {analysis.get('main_subject', 'not identified')}")
                return analysis
            else:
                print("   ⚠️ LLM response format error, using fallback")
                return self._fallback_llm_analysis(question)
                
        except Exception as e:
            print(f"   ⚠️ LLM analysis error: {e}")
            return self._fallback_llm_analysis(question)

    def _build_keyword_graph(self, question: str, nlp_analysis: Dict, llm_analysis: Dict):
        """Build a graph representation of keywords and their relationships"""
        if not ADVANCED_NLP_AVAILABLE:
            return None
            
        G = nx.Graph()
        
        # Add nodes from spaCy entities
        for entity in nlp_analysis["entities"]:
            G.add_node(entity["text"].lower(), 
                      node_type="entity",
                      entity_label=entity["label"],
                      importance=self._calculate_entity_importance(entity),
                      source="spacy")
        
        # Add nodes from LLM analysis
        main_subject = llm_analysis.get("main_subject", "")
        if main_subject:
            G.add_node(main_subject.lower(),
                      node_type="main_subject", 
                      importance=1.0,
                      source="llm")
        
        for concept in llm_analysis.get("key_concepts", []):
            G.add_node(concept.lower(),
                      node_type="concept",
                      importance=0.8,
                      source="llm")
        
        # Add nodes from noun phrases
        for phrase in nlp_analysis["noun_phrases"]:
            if phrase not in G.nodes:
                G.add_node(phrase,
                          node_type="noun_phrase",
                          importance=0.6,
                          source="spacy")
        
        # Add edges based on relationships
        self._add_graph_relationships(G, nlp_analysis, llm_analysis, question)
        
        # Calculate centrality measures
        if len(G.nodes) > 1:
            centrality = nx.degree_centrality(G)
            nx.set_node_attributes(G, centrality, "centrality")
        
        return G

    def _add_graph_relationships(self, G, nlp_analysis: Dict, llm_analysis: Dict, question: str):
        """Add relationships between nodes in the graph"""
        if not G:
            return
            
        # Add relationships from spaCy dependencies
        for dep in nlp_analysis["dependencies"]:
            dependent = dep["dependent"].lower()
            head = dep["head"].lower()
            relation = dep["relation"]
            
            if dependent in G.nodes and head in G.nodes:
                G.add_edge(dependent, head,
                          relation=relation,
                          weight=0.6,
                          source="spacy")
        
        # Add proximity relationships (words appearing close in text)
        question_words = question.lower().split()
        for i, word1 in enumerate(question_words):
            for j, word2 in enumerate(question_words):
                if (abs(i - j) <= 3 and word1 in G.nodes and word2 in G.nodes 
                    and word1 != word2 and not G.has_edge(word1, word2)):
                    G.add_edge(word1, word2,
                              relation="proximity",
                              weight=0.4,
                              source="proximity")

    def _extract_final_keywords(self, G, max_keywords: int) -> List[str]:
        """Extract final keywords using graph analysis"""
        if not G or not G.nodes:
            return []
        
        # Score nodes based on multiple factors
        node_scores = {}
        for node in G.nodes:
            data = G.nodes[node]
            
            # Base importance
            score = data.get("importance", 0.5)
            
            # Centrality bonus
            score += data.get("centrality", 0) * 0.3
            
            # Node type bonus
            if data.get("node_type") == "main_subject":
                score += 0.4
            elif data.get("node_type") == "entity":
                score += 0.3
            elif data.get("node_type") == "concept":
                score += 0.2
            
            # Length penalty for very long phrases
            if len(node) > 20:
                score *= 0.8
            
            node_scores[node] = score
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        final_keywords = [kw for kw, score in sorted_keywords[:max_keywords]]
        
        print(f"   📊 Top keyword scores: {dict(sorted_keywords[:5])}")
        return final_keywords

    def _generate_search_variants(self, keywords: List[str], question: str) -> List[str]:
        """Generate search variants using synonyms and domain knowledge"""
        variants = set(keywords)  # Start with original keywords
        
        # Add combinations
        if len(keywords) >= 2:
            for i, kw1 in enumerate(keywords):
                for kw2 in keywords[i+1:]:
                    if len(f"{kw1} {kw2}") <= 30:  # Reasonable length limit
                        variants.add(f"{kw1} {kw2}")
        
        # Add domain-specific expansions
        domain = self._detect_domain(question, keywords)
        if domain in self.domain_entities:
            for keyword in keywords[:3]:  # Only expand top 3
                for domain_term in self.domain_entities[domain][:2]:  # Top 2 domain terms
                    if domain_term not in keyword.lower():
                        variants.add(f"{keyword} {domain_term}")
        
        return list(variants)[:15]  # Limit total variants

    def _calculate_entity_importance(self, entity: Dict) -> float:
        """Calculate importance score for an entity"""
        base_score = 0.7
        
        # Entity type importance
        type_scores = {
            "PERSON": 0.9, "ORG": 0.9, "GPE": 0.8,  # High importance
            "MONEY": 0.8, "DATE": 0.6, "TIME": 0.5,  # Medium importance
            "CARDINAL": 0.4, "ORDINAL": 0.4           # Lower importance
        }
        
        return type_scores.get(entity.get("label", ""), base_score)

    def _detect_domain(self, question: str, keywords: List[str]) -> str:
        """Detect the domain of the question"""
        question_lower = question.lower()
        all_text = f"{question_lower} {' '.join(keywords)}"
        
        domain_indicators = {
            "education": ["university", "college", "harvard", "endowment", "tuition", "academic"],
            "finance": ["investment", "fund", "market", "economy", "stock", "financial"],
            "politics": ["government", "policy", "election", "trump", "biden", "congress"],
            "technology": ["ai", "tech", "software", "innovation", "startup", "digital"],
            "health": ["healthcare", "medical", "hospital", "pharmaceutical", "health"]
        }
        
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in all_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general"

    def _analyze_graph(self, G) -> Dict[str, Any]:
        """Analyze the keyword graph structure"""
        if not G or not G.nodes:
            return {"nodes": 0, "edges": 0, "components": 0}
        
        return {
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "components": nx.number_connected_components(G),
            "density": nx.density(G),
            "most_central": max(G.nodes, key=lambda n: G.nodes[n].get("centrality", 0)) if G.nodes else None
        }

    def _calculate_extraction_confidence(self, nlp_analysis: Dict, llm_analysis: Dict) -> float:
        """Calculate confidence in the keyword extraction"""
        confidence = 0.5  # Base confidence
        
        # Boost for entities found
        if nlp_analysis["entities"]:
            confidence += 0.2
        
        # Boost for LLM main subject
        if llm_analysis.get("main_subject"):
            confidence += 0.2
        
        # Boost for noun phrases
        if len(nlp_analysis["noun_phrases"]) >= 2:
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _fallback_llm_analysis(self, question: str) -> Dict[str, Any]:
        """Fallback analysis if LLM JSON parsing fails"""
        return {
            "main_subject": "",
            "key_concepts": [],
            "search_focus": "general information",
            "domain": "general",
            "relationships": []
        }

    def debug_graph_formation(self, question: str) -> Dict[str, Any]:
        """Debug method to show graph formation step-by-step"""
        print(f"\n🔬 DEBUGGING GRAPH FORMATION FOR: '{question}'")
        print("-" * 50)
        
        try:
            # Test basic functionality
            result = self.extract_enhanced_keywords(question, max_keywords=6)
            
            print(f"\n🎉 **EXTRACTION DEBUG SUCCESSFUL!**")
            print(f"   • Keywords: {result['primary_keywords']}")
            print(f"   • Variants: {result['search_variants'][:5]}")
            print(f"   • Confidence: {result['confidence']:.1%}")
            
            return {
                "success": True,
                "keywords": result['primary_keywords'],
                "variants": result['search_variants'],
                "confidence": result['confidence']
            }
            
        except Exception as e:
            print(f"❌ Debug failed: {e}")
            return {"success": False, "error": str(e)}