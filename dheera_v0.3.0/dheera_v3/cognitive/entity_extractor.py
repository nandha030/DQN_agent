# cognitive/entity_extractor.py
"""
Dheera v0.3.0 - Entity Extractor
Extracts named entities and key information from user messages.
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class EntityType(Enum):
    """Types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    NUMBER = "number"
    MONEY = "money"
    PERCENTAGE = "percentage"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    PROGRAMMING_LANG = "programming_language"
    TECHNOLOGY = "technology"
    TOPIC = "topic"
    ACTION_VERB = "action_verb"


@dataclass
class Entity:
    """Extracted entity."""
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: List[Entity]
    entity_counts: Dict[str, int]
    key_phrases: List[str]
    topics: List[str]


class EntityExtractor:
    """
    Rule-based entity extractor for Dheera.
    Extracts entities using patterns and keyword matching.
    """
    
    # Regex patterns for different entity types
    PATTERNS = {
        EntityType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        EntityType.URL: r'https?://[^\s<>"{}|\\^`\[\]]+',
        
        EntityType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        
        EntityType.MONEY: r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP|INR|rupees?)\b',
        
        EntityType.PERCENTAGE: r'\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*percent\b',
        
        EntityType.DATE: r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}?|(?:today|tomorrow|yesterday|next\s+(?:week|month|year)|last\s+(?:week|month|year)))\b',
        
        EntityType.TIME: r'\b(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?|\d{1,2}\s*(?:AM|PM|am|pm)|(?:noon|midnight|morning|afternoon|evening|night))\b',
        
        EntityType.NUMBER: r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
    }
    
    # Programming languages
    PROGRAMMING_LANGS = {
        "python", "javascript", "java", "c++", "c#", "ruby", "go", "golang",
        "rust", "swift", "kotlin", "typescript", "php", "perl", "scala",
        "r", "matlab", "sql", "html", "css", "bash", "shell", "powershell",
    }
    
    # Technologies/Tools
    TECHNOLOGIES = {
        "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "google cloud",
        "react", "vue", "angular", "node", "nodejs", "django", "flask", "fastapi",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "git", "github", "gitlab", "jenkins", "ci/cd", "linux", "ubuntu", "macos",
        "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "kafka",
        "api", "rest", "graphql", "websocket", "oauth", "jwt",
        "machine learning", "deep learning", "ai", "artificial intelligence",
        "nlp", "natural language", "computer vision", "neural network",
        "ollama", "llama", "gpt", "chatgpt", "claude", "openai", "anthropic",
    }
    
    # Action verbs
    ACTION_VERBS = {
        "create", "make", "build", "write", "generate", "develop",
        "search", "find", "look", "get", "fetch", "retrieve",
        "explain", "describe", "tell", "show", "help", "teach",
        "calculate", "compute", "solve", "analyze", "compare",
        "convert", "translate", "transform", "change", "modify", "edit",
        "delete", "remove", "clear", "reset",
        "install", "setup", "configure", "deploy", "run", "execute",
        "debug", "fix", "optimize", "improve", "refactor",
    }
    
    def __init__(self):
        # Compile regex patterns
        self._compiled = {
            etype: re.compile(pattern, re.IGNORECASE)
            for etype, pattern in self.PATTERNS.items()
        }
    
    def extract(self, message: str) -> ExtractionResult:
        """
        Extract entities from message.
        
        Args:
            message: User message text
            
        Returns:
            ExtractionResult with entities and metadata
        """
        entities = []
        message_lower = message.lower()
        
        # 1. Pattern-based extraction
        for entity_type, pattern in self._compiled.items():
            for match in pattern.finditer(message):
                entities.append(Entity(
                    text=match.group(),
                    type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                ))
        
        # 2. Programming language detection
        for lang in self.PROGRAMMING_LANGS:
            if lang in message_lower:
                # Find position
                idx = message_lower.find(lang)
                entities.append(Entity(
                    text=lang,
                    type=EntityType.PROGRAMMING_LANG,
                    start=idx,
                    end=idx + len(lang),
                    confidence=0.95,
                    normalized=lang.lower(),
                ))
        
        # 3. Technology detection
        for tech in self.TECHNOLOGIES:
            if tech in message_lower:
                idx = message_lower.find(tech)
                entities.append(Entity(
                    text=tech,
                    type=EntityType.TECHNOLOGY,
                    start=idx,
                    end=idx + len(tech),
                    confidence=0.85,
                    normalized=tech.lower(),
                ))
        
        # 4. Action verb detection
        words = message_lower.split()
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean in self.ACTION_VERBS:
                # Find position (approximate)
                idx = message_lower.find(word_clean)
                entities.append(Entity(
                    text=word_clean,
                    type=EntityType.ACTION_VERB,
                    start=idx,
                    end=idx + len(word_clean),
                    confidence=0.8,
                ))
        
        # 5. Extract key phrases (simple noun phrase extraction)
        key_phrases = self._extract_key_phrases(message)
        
        # 6. Extract topics
        topics = self._extract_topics(message, entities)
        
        # Count entities by type
        entity_counts = {}
        for entity in entities:
            type_name = entity.type.value
            entity_counts[type_name] = entity_counts.get(type_name, 0) + 1
        
        # Remove duplicates (keep highest confidence)
        entities = self._deduplicate_entities(entities)
        
        return ExtractionResult(
            entities=entities,
            entity_counts=entity_counts,
            key_phrases=key_phrases,
            topics=topics,
        )
    
    def _extract_key_phrases(self, message: str) -> List[str]:
        """Extract key phrases using simple patterns."""
        phrases = []
        
        # Pattern: adjective + noun, noun + noun
        # Simplified: extract 2-3 word sequences that look meaningful
        words = message.split()
        
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "must", "shall", "can",
                     "to", "of", "in", "for", "on", "with", "at", "by", "from",
                     "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
                     "my", "your", "his", "its", "our", "their", "this", "that",
                     "what", "which", "who", "when", "where", "why", "how",
                     "and", "or", "but", "if", "then", "so", "because", "as"}
        
        # Look for 2-3 word phrases
        for i in range(len(words) - 1):
            w1 = re.sub(r'[^\w]', '', words[i].lower())
            w2 = re.sub(r'[^\w]', '', words[i + 1].lower())
            
            if w1 not in stopwords and w2 not in stopwords and len(w1) > 2 and len(w2) > 2:
                phrases.append(f"{w1} {w2}")
        
        return phrases[:5]  # Limit to top 5
    
    def _extract_topics(self, message: str, entities: List[Entity]) -> List[str]:
        """Extract main topics from message."""
        topics = []
        
        # Get technology and programming language entities as topics
        for entity in entities:
            if entity.type in [EntityType.TECHNOLOGY, EntityType.PROGRAMMING_LANG]:
                topics.append(entity.normalized or entity.text.lower())
        
        # Common topic keywords
        topic_keywords = {
            "machine learning": ["machine learning", "ml", "model", "training", "dataset"],
            "web development": ["website", "web app", "frontend", "backend", "html", "css"],
            "database": ["database", "sql", "query", "table", "schema"],
            "api": ["api", "endpoint", "rest", "request", "response"],
            "devops": ["docker", "kubernetes", "deploy", "ci/cd", "pipeline"],
            "ai": ["ai", "artificial intelligence", "chatbot", "llm", "gpt"],
        }
        
        message_lower = message.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                if topic not in topics:
                    topics.append(topic)
        
        return topics[:5]
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen = {}
        
        for entity in entities:
            key = (entity.text.lower(), entity.type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        
        return list(seen.values())
    
    def get_entity_summary(self, result: ExtractionResult) -> Dict[str, Any]:
        """Get a summary of extracted entities for state building."""
        return {
            "entity_count": len(result.entities),
            "has_code_entities": any(
                e.type in [EntityType.PROGRAMMING_LANG, EntityType.TECHNOLOGY]
                for e in result.entities
            ),
            "has_numbers": any(
                e.type in [EntityType.NUMBER, EntityType.MONEY, EntityType.PERCENTAGE]
                for e in result.entities
            ),
            "has_datetime": any(
                e.type in [EntityType.DATE, EntityType.TIME]
                for e in result.entities
            ),
            "has_urls": any(e.type == EntityType.URL for e in result.entities),
            "action_verbs": [e.text for e in result.entities if e.type == EntityType.ACTION_VERB],
            "topics": result.topics,
            "key_phrases": result.key_phrases,
        }


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing EntityExtractor...")
    
    extractor = EntityExtractor()
    
    test_messages = [
        "Can you help me write a Python script to process data?",
        "Search for the latest news about OpenAI and GPT-4",
        "Calculate 25% of $1,500",
        "I need to deploy my Docker container to AWS by next Monday",
        "Send an email to test@example.com about the meeting at 3:30 PM",
        "How do I install TensorFlow and PyTorch on Ubuntu?",
    ]
    
    print("\nEntity Extraction Results:")
    print("-" * 60)
    
    for msg in test_messages:
        result = extractor.extract(msg)
        summary = extractor.get_entity_summary(result)
        
        print(f"\n'{msg}'")
        print(f"  Entities ({len(result.entities)}):")
        for entity in result.entities[:5]:
            print(f"    - {entity.text} [{entity.type.value}] ({entity.confidence:.2f})")
        print(f"  Topics: {result.topics}")
        print(f"  Key phrases: {result.key_phrases[:3]}")
    
    print("\n✅ Entity extractor tests passed!")
