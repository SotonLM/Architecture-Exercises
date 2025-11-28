"""
Miniature Semantic Search Engine
Demonstrates embedding-based document retrieval with performance metrics
"""

import numpy as np
import json
import time
from typing import List, Tuple, Dict
from pathlib import Path

# Corpus spanning 4 topics: Space, Cooking, History, Technology
CORPUS = [
    # Space exploration (docs 0-6)
    "The James Webb Space Telescope orbits the Sun at the second Lagrange point, about 1.5 million kilometers from Earth. It uses infrared light to peer through cosmic dust.",
    "Mars has two small moons named Phobos and Deimos. Scientists believe they may be captured asteroids due to their irregular shapes and composition.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse.",
    "The International Space Station travels at 28,000 kilometers per hour, completing an orbit around Earth every 90 minutes.",
    "Saturn's rings are made mostly of ice particles, with sizes ranging from tiny grains to house-sized chunks. The rings are incredibly thin relative to their diameter.",
    "Exoplanets are planets that orbit stars outside our solar system. Over 5,000 have been confirmed, with many in the habitable zone.",
    "The Voyager 1 spacecraft, launched in 1977, is now in interstellar space and is the most distant human-made object.",
    
    # Cooking (docs 7-13)
    "Maillard reaction occurs when proteins and sugars in food are heated, creating complex flavors and brown colors. This is essential for searing meat.",
    "Sourdough bread relies on wild yeast and bacteria for fermentation, creating a tangy flavor and chewy texture. The starter must be fed regularly.",
    "Emulsification combines oil and water-based ingredients using emulsifiers like egg yolk or mustard. Mayonnaise is a classic example.",
    "Caramelization happens when sugar is heated above 160°C, breaking down into hundreds of compounds that create rich, complex sweetness.",
    "Sous vide cooking involves sealing food in bags and cooking in precisely controlled water baths, ensuring even temperature throughout.",
    "Knife skills are foundational in cooking. The proper grip, claw technique, and consistent cuts improve safety and cooking uniformity.",
    "Umami is the fifth basic taste, triggered by glutamates. It's abundant in aged cheeses, tomatoes, mushrooms, and fermented foods.",
    
    # History (docs 14-21)
    "The Library of Alexandria was one of the largest and most significant libraries of the ancient world. Its destruction represents a major loss of knowledge.",
    "The Silk Road was an ancient network of trade routes connecting East and West, facilitating not just commerce but cultural exchange.",
    "The printing press, invented by Johannes Gutenberg around 1440, revolutionized information dissemination and literacy in Europe.",
    "The fall of Constantinople in 1453 marked the end of the Byzantine Empire and led many Greek scholars to flee to Western Europe.",
    "The Industrial Revolution began in Britain in the late 18th century, transforming manufacturing through mechanization and factory systems.",
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and established the modern concept of national sovereignty.",
    "The Code of Hammurabi, created around 1750 BCE, is one of the oldest deciphered writings of significant length and established legal precedents.",
    "The Viking Age lasted from roughly 793 to 1066 CE. Norse explorers reached North America centuries before Columbus.",
    
    # Technology (docs 22-29)
    "Machine learning models learn patterns from data without explicit programming. Neural networks are inspired by biological brain structure.",
    "Quantum computers use qubits that can exist in superposition, potentially solving certain problems exponentially faster than classical computers.",
    "Blockchain technology creates immutable distributed ledgers through cryptographic hashing and consensus mechanisms.",
    "The transistor, invented in 1947, is the fundamental building block of modern electronics. Billions fit on a single microchip.",
    "5G networks use higher frequency bands and advanced antenna technology to achieve lower latency and faster data speeds.",
    "CRISPR gene editing allows precise modification of DNA sequences, with applications in medicine, agriculture, and research.",
    "Fiber optic cables transmit data as light pulses through glass fibers, enabling high-speed internet across continents.",
    "The Internet of Things connects everyday devices to the internet, enabling smart homes, wearables, and industrial automation.",
]

EMBEDDING_CACHE_FILE = "corpus_embeddings.npy"
METADATA_CACHE_FILE = "corpus_metadata.json"


def simple_embedding(text: str, dim: int = 100) -> np.ndarray:
    """
    Simple deterministic embedding using character-level hashing.
    In production, use sentence-transformers, OpenAI API, or similar.
    """
    # Normalize text
    text = text.lower().strip()
    
    # Create embedding vector
    embedding = np.zeros(dim)
    
    # Word-level features
    words = text.split()
    for i, word in enumerate(words):
        # Hash word to dimension space
        word_hash = hash(word) % dim
        embedding[word_hash] += 1.0
        
        # Bigram features
        if i < len(words) - 1:
            bigram = f"{word}_{words[i+1]}"
            bigram_hash = hash(bigram) % dim
            embedding[bigram_hash] += 0.5
    
    # Character-level features for robustness
    for i, char in enumerate(text):
        if char.isalnum():
            char_hash = (hash(char) + i) % dim
            embedding[char_hash] += 0.1
    
    # Normalize to unit vector
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


class SemanticSearchEngine:
    def __init__(self, corpus: List[str], embedding_dim: int = 100):
        self.corpus = corpus
        self.embedding_dim = embedding_dim
        self.embeddings = None
        
    def build_index(self, use_cache: bool = True) -> float:
        """
        Compute embeddings for entire corpus.
        Returns time taken in seconds.
        """
        cache_path = Path(EMBEDDING_CACHE_FILE)
        metadata_path = Path(METADATA_CACHE_FILE)
        
        # Try to load from cache
        if use_cache and cache_path.exists() and metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify cache is valid
            if (metadata['corpus_size'] == len(self.corpus) and 
                metadata['embedding_dim'] == self.embedding_dim):
                print("Loading embeddings from cache...")
                start = time.time()
                self.embeddings = np.load(cache_path)
                elapsed = time.time() - start
                print(f"Loaded {len(self.embeddings)} embeddings in {elapsed:.4f}s")
                return elapsed
        
        # Compute embeddings
        print("Computing embeddings for corpus...")
        start = time.time()
        
        self.embeddings = np.array([
            simple_embedding(doc, self.embedding_dim) 
            for doc in self.corpus
        ])
        
        elapsed = time.time() - start
        print(f"Computed {len(self.embeddings)} embeddings in {elapsed:.4f}s")
        
        # Save to cache
        np.save(cache_path, self.embeddings)
        with open(metadata_path, 'w') as f:
            json.dump({
                'corpus_size': len(self.corpus),
                'embedding_dim': self.embedding_dim,
                'timestamp': time.time()
            }, f)
        print(f"Saved embeddings to {cache_path}")
        
        return elapsed
    
    def search(self, query: str, k: int = 3) -> Tuple[List[Dict], float]:
        """
        Search for top-k most similar documents.
        Returns (results, query_time).
        """
        if self.embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        start = time.time()
        
        # Embed query
        query_embedding = simple_embedding(query, self.embedding_dim)
        
        # Compute similarities
        similarities = np.array([
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self.embeddings
        ])
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Format results
        results = []
        for idx in top_k_indices:
            doc = self.corpus[idx]
            # Create snippet (first 100 chars)
            snippet = doc if len(doc) <= 100 else doc[:97] + "..."
            
            results.append({
                'doc_id': int(idx),
                'score': float(similarities[idx]),
                'snippet': snippet,
                'full_text': doc
            })
        
        query_time = time.time() - start
        
        return results, query_time


def display_results(query: str, results: List[Dict], query_time: float):
    """Display search results in a formatted way."""
    print(f"\n{'='*80}")
    print(f"Query: \"{query}\"")
    print(f"Query time: {query_time*1000:.2f}ms")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        print(f"#{i} [Doc {result['doc_id']}] (Score: {result['score']:.4f})")
        print(f"   {result['snippet']}")
        print()


def run_benchmark_queries(engine: SemanticSearchEngine):
    """Run predefined diverse queries to test retrieval quality."""
    
    test_queries = [
        # Factual, specific
        ("black holes and gravity", "Space - specific scientific concept"),
        
        # Vague, multi-topic potential
        ("ancient knowledge", "History - could match multiple documents"),
        
        # Long, detailed query
        ("how does heating sugar create sweet flavors and brown colors in cooking", "Cooking - detailed multi-concept"),
        
        # Short, single word
        ("fermentation", "Short - should match cooking/biology"),
        
        # Cross-topic conceptual
        ("speed and distance measurements", "Multi-topic - could match space or tech"),
        
        # Technology specific
        ("computers processing information", "Technology - computing concepts"),
        
        # Historical trade
        ("trade routes and cultural exchange", "History - Silk Road specific"),
    ]
    
    print("\n" + "="*80)
    print("RUNNING BENCHMARK QUERIES")
    print("="*80)
    
    total_time = 0
    
    for query, description in test_queries:
        print(f"\nTest: {description}")
        results, query_time = engine.search(query, k=3)
        display_results(query, results, query_time)
        total_time += query_time
        
        # Sanity check
        print("Sanity check:")
        if "black holes" in query.lower():
            if any("black hole" in r['full_text'].lower() for r in results[:2]):
                print("   Correctly retrieved black hole document")
            else:
                print("   May have missed black hole document")
        
        if "fermentation" in query.lower():
            if any("ferment" in r['full_text'].lower() for r in results[:2]):
                print("   Found fermentation-related document")
            else:
                print("   Fermentation match quality unclear")
    
    print(f"\n{'='*80}")
    print(f"Total benchmark time: {total_time*1000:.2f}ms")
    print(f"Average query time: {(total_time/len(test_queries))*1000:.2f}ms")
    print(f"{'='*80}\n")


def main():
    """Main execution flow."""
    print("="*80)
    print("SEMANTIC SEARCH ENGINE")
    print("="*80)
    print(f"\nCorpus: {len(CORPUS)} documents")
    print(f"   - Space exploration: 7 docs")
    print(f"   - Cooking: 7 docs")
    print(f"   - History: 8 docs")
    print(f"   - Technology: 8 docs")
    print()
    
    # Initialize engine
    engine = SemanticSearchEngine(CORPUS)
    
    # Build index and time it
    index_time = engine.build_index(use_cache=True)
    
    print(f"\nPerformance Summary:")
    print(f"   Corpus indexing: {index_time:.4f}s ({index_time*1000:.2f}ms)")
    print(f"   Per-document: {(index_time/len(CORPUS))*1000:.2f}ms")
    print()
    
    # Run automated benchmark
    run_benchmark_queries(engine)
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter your search queries (or 'quit' to exit)\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            results, query_time = engine.search(query, k=3)
            display_results(query, results, query_time)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()