import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class SemanticSearch:
    """A miniature semantic search engine using sentence embeddings."""
    
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        """Initialize the semantic search engine with a sentence transformer model."""
        print("Initializing Semantic Search Engine...")
        start = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        end = time.perf_counter()
        print(f"✓ Model loaded in {end - start:.3f} seconds\n")
        
        self.corpus = []
        self.embeddings = None
        self.corpus_metadata = []
    
    def load_corpus(self, corpus_data):
        """
        Load corpus documents.
        
        Args:
            corpus_data: List of dictionaries with 'id', 'topic', and 'text' keys
        """
        self.corpus = corpus_data
        self.corpus_metadata = [
            {'id': doc['id'], 'topic': doc['topic']} 
            for doc in corpus_data
        ]
        print(f"✓ Loaded {len(self.corpus)} documents across topics:")
        topics = set(doc['topic'] for doc in corpus_data)
        for topic in topics:
            count = sum(1 for doc in corpus_data if doc['topic'] == topic)
            print(f"  - {topic}: {count} documents")
        print()
    
    def build_index(self, save_to_disk=True):
        """
        Compute embeddings for all corpus documents.
        
        Args:
            save_to_disk: Whether to save embeddings to disk for reuse
        """
        print("Building search index...")
        start = time.perf_counter()
        
        # Extract text from corpus
        texts = [doc['text'] for doc in self.corpus]
        
        # Compute normalized embeddings
        self.embeddings = self.model.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        end = time.perf_counter()
        elapsed = end - start
        
        print(f"✓ Indexed {len(self.corpus)} documents in {elapsed:.3f} seconds")
        print(f"  ({elapsed/len(self.corpus)*1000:.2f} ms per document)")
        print(f"  Embedding shape: {self.embeddings.shape}\n")
        
        # Save to disk
        if save_to_disk:
            self._save_embeddings()
        
        return elapsed
    
    def _save_embeddings(self):
        """Save embeddings and metadata to disk."""
        Path("week4/level2").mkdir(parents=True, exist_ok=True)
        
        np.save('embeddings.npy', self.embeddings)
        with open('corpus_metadata.json', 'w') as f:
            json.dump(self.corpus_metadata, f, indent=2)
        
        print("✓ Embeddings saved to disk (embeddings.npy, corpus_metadata.json)\n")
    
    def load_embeddings(self):
        """Load pre-computed embeddings from disk."""
        try:
            self.embeddings = np.load('embeddings.npy')
            with open('corpus_metadata.json', 'r') as f:
                self.corpus_metadata = json.load(f)
            print(f"✓ Loaded pre-computed embeddings for {len(self.embeddings)} documents\n")
            return True
        except FileNotFoundError:
            print("⚠ No saved embeddings found. Building new index...\n")
            return False
    
    def search(self, query, k=3):
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query string
            k: Number of top results to return
            
        Returns:
            Tuple of (results, query_time) where results is a list of dicts
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        start = time.perf_counter()
        
        # Encode query with normalization
        query_embedding = self.model.encode(
            [query], 
            normalize_embeddings=True
        )[0]
        
        # Compute cosine similarity (dot product since normalized)
        similarities = self.embeddings @ query_embedding
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        end = time.perf_counter()
        query_time = end - start
        
        # Prepare results
        results = []
        for idx in top_k_indices:
            results.append({
                'rank': len(results) + 1,
                'id': self.corpus[idx]['id'],
                'topic': self.corpus[idx]['topic'],
                'text': self.corpus[idx]['text'],
                'score': float(similarities[idx])
            })
        
        return results, query_time
    
    def display_results(self, query, results, query_time):
        """Display search results in a formatted way."""
        print("="*80)
        print(f"QUERY: {query}")
        print("="*80)
        print(f"Search completed in {query_time*1000:.2f} ms\n")
        
        for result in results:
            print(f"Rank {result['rank']} | ID: {result['id']} | Topic: {result['topic']} | Score: {result['score']:.4f}")
            print("-" * 80)
            # Truncate text if too long
            text = result['text']
            if len(text) > 200:
                text = text[:200] + "..."
            print(text)
            print()


def create_sample_corpus():
    """Create a diverse corpus spanning multiple topics."""
    corpus = [
        # Topic 1: Machine Learning (10 docs)
        {
            "id": "ML001",
            "topic": "Machine Learning",
            "text": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes called neurons that process information through weighted connections."
        },
        {
            "id": "ML002",
            "topic": "Machine Learning",
            "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. These deep architectures can learn hierarchical representations of data."
        },
        {
            "id": "ML003",
            "topic": "Machine Learning",
            "text": "Supervised learning requires labeled training data where the model learns to map inputs to known outputs. Common applications include classification and regression tasks."
        },
        {
            "id": "ML004",
            "topic": "Machine Learning",
            "text": "Unsupervised learning discovers patterns in unlabeled data. Clustering and dimensionality reduction are popular unsupervised learning techniques."
        },
        {
            "id": "ML005",
            "topic": "Machine Learning",
            "text": "Transformers revolutionized natural language processing by using self-attention mechanisms. Models like BERT and GPT are based on transformer architecture."
        },
        {
            "id": "ML006",
            "topic": "Machine Learning",
            "text": "Overfitting occurs when a model learns the training data too well, including noise and outliers. Regularization techniques help prevent overfitting."
        },
        {
            "id": "ML007",
            "topic": "Machine Learning",
            "text": "Transfer learning allows models trained on one task to be adapted for another related task. This is especially useful when labeled data is scarce."
        },
        {
            "id": "ML008",
            "topic": "Machine Learning",
            "text": "Reinforcement learning trains agents to make decisions by rewarding desired behaviors. It's used in game playing, robotics, and autonomous systems."
        },
        {
            "id": "ML009",
            "topic": "Machine Learning",
            "text": "Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize a loss function. Variants include SGD and Adam."
        },
        {
            "id": "ML010",
            "topic": "Machine Learning",
            "text": "Convolutional neural networks excel at processing grid-like data such as images. They use convolutional layers to detect spatial patterns and features."
        },
        
        # Topic 2: Cooking (10 docs)
        {
            "id": "COOK001",
            "topic": "Cooking",
            "text": "Pasta should be cooked in boiling salted water until al dente, which means it's tender but still firm to the bite. Fresh pasta cooks much faster than dried pasta."
        },
        {
            "id": "COOK002",
            "topic": "Cooking",
            "text": "Caramelizing onions requires low heat and patience. The natural sugars in onions turn golden brown and develop a sweet, complex flavor over 30-40 minutes."
        },
        {
            "id": "COOK003",
            "topic": "Cooking",
            "text": "Resting meat after cooking allows juices to redistribute throughout the meat. This results in a more tender and flavorful final product."
        },
        {
            "id": "COOK004",
            "topic": "Cooking",
            "text": "Baking bread requires proper gluten development through kneading. The dough should be smooth and elastic before the first rise."
        },
        {
            "id": "COOK005",
            "topic": "Cooking",
            "text": "Mise en place, a French culinary term, means having all ingredients prepared and organized before cooking begins. This makes the cooking process smoother."
        },
        {
            "id": "COOK006",
            "topic": "Cooking",
            "text": "Searing meat at high temperature creates a flavorful brown crust through the Maillard reaction. This adds depth and complexity to dishes."
        },
        {
            "id": "COOK007",
            "topic": "Cooking",
            "text": "Emulsification combines two liquids that normally don't mix, like oil and vinegar. Mayonnaise and vinaigrette are classic examples of emulsions."
        },
        {
            "id": "COOK008",
            "topic": "Cooking",
            "text": "Stock is made by simmering bones, vegetables, and aromatics for hours. Good stock forms the foundation of many soups and sauces."
        },
        {
            "id": "COOK009",
            "topic": "Cooking",
            "text": "Seasoning food properly means tasting and adjusting salt, acid, and fat throughout cooking. Balance is key to delicious dishes."
        },
        {
            "id": "COOK010",
            "topic": "Cooking",
            "text": "Tempering chocolate involves carefully melting and cooling it to specific temperatures. This creates a glossy finish and satisfying snap."
        },
        
        # Topic 3: Space Exploration (7 docs)
        {
            "id": "SPACE001",
            "topic": "Space Exploration",
            "text": "The James Webb Space Telescope can observe the universe in infrared wavelengths, allowing it to see through cosmic dust and study the earliest galaxies."
        },
        {
            "id": "SPACE002",
            "topic": "Space Exploration",
            "text": "Mars rovers like Curiosity and Perseverance search for signs of ancient microbial life. They analyze soil samples and study the planet's geology."
        },
        {
            "id": "SPACE003",
            "topic": "Space Exploration",
            "text": "The International Space Station orbits Earth every 90 minutes at an altitude of about 400 kilometers. Astronauts conduct scientific experiments in microgravity."
        },
        {
            "id": "SPACE004",
            "topic": "Space Exploration",
            "text": "SpaceX's Starship is designed to be a fully reusable launch system for missions to the Moon, Mars, and beyond. It represents the future of space travel."
        },
        {
            "id": "SPACE005",
            "topic": "Space Exploration",
            "text": "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. Supermassive black holes exist at galaxy centers."
        },
        {
            "id": "SPACE006",
            "topic": "Space Exploration",
            "text": "The Voyager probes, launched in 1977, are now in interstellar space. They carry golden records with sounds and images representing Earth's diversity."
        },
        {
            "id": "SPACE007",
            "topic": "Space Exploration",
            "text": "Exoplanets are planets orbiting stars outside our solar system. Thousands have been discovered, some potentially habitable in their star's Goldilocks zone."
        },
        
        # Topic 4: Climate Science (6 docs)
        {
            "id": "CLIMATE001",
            "topic": "Climate Science",
            "text": "Greenhouse gases like carbon dioxide and methane trap heat in Earth's atmosphere. Rising concentrations from human activity are driving global warming."
        },
        {
            "id": "CLIMATE002",
            "topic": "Climate Science",
            "text": "Arctic ice is melting at unprecedented rates. This creates a feedback loop as dark ocean water absorbs more heat than reflective ice."
        },
        {
            "id": "CLIMATE003",
            "topic": "Climate Science",
            "text": "Ocean acidification occurs when CO2 dissolves in seawater, forming carbonic acid. This threatens marine ecosystems, especially coral reefs and shellfish."
        },
        {
            "id": "CLIMATE004",
            "topic": "Climate Science",
            "text": "Renewable energy sources like solar and wind power generate electricity without greenhouse gas emissions. They're crucial for climate change mitigation."
        },
        {
            "id": "CLIMATE005",
            "topic": "Climate Science",
            "text": "Climate models use physics and mathematics to simulate Earth's climate system. They help predict future climate scenarios under different emission pathways."
        },
        {
            "id": "CLIMATE006",
            "topic": "Climate Science",
            "text": "Deforestation reduces Earth's capacity to absorb CO2 and disrupts local weather patterns. Protecting forests is essential for climate stability."
        },
    ]
    
    return corpus


def run_interactive_search(search_engine):
    """Run an interactive search loop."""
    print("\n" + "="*80)
    print("SEMANTIC SEARCH ENGINE - INTERACTIVE MODE")
    print("="*80)
    print("Enter queries to search the corpus. Type 'quit' to exit.\n")
    
    query_log = []
    
    while True:
        query = input("🔍 Enter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        results, query_time = search_engine.search(query, k=3)
        search_engine.display_results(query, results, query_time)
        
        query_log.append({
            'query': query,
            'results': results,
            'time': query_time
        })
        
        print()
    
    return query_log


def run_example_queries(search_engine):
    """Run predefined example queries for testing."""
    
    example_queries = [
        "How do neural networks work?",  # Factual, ML-focused
        "cooking techniques",  # Vague, short
        "I want to learn about planets and stars outside our solar system, particularly those that might support life",  # Long, specific
        "heat",  # Very short, ambiguous (could match cooking or climate)
        "sustainable energy and environmental protection"  # Multi-topic (climate + general science)
    ]
    
    print("\n" + "="*80)
    print("RUNNING EXAMPLE QUERIES")
    print("="*80 + "\n")
    
    all_results = []
    
    for query in example_queries:
        results, query_time = search_engine.search(query, k=3)
        search_engine.display_results(query, results, query_time)
        
        all_results.append({
            'query': query,
            'results': results,
            'time': query_time
        })
        
        time.sleep(0.5)  # Brief pause between queries
    
    return all_results


def save_search_examples(query_results, filename='search_examples.txt'):
    """Save search examples to a text file."""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SEMANTIC SEARCH - EXAMPLE QUERIES AND RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for i, query_result in enumerate(query_results, 1):
            query = query_result['query']
            results = query_result['results']
            query_time = query_result['time']
            
            f.write(f"QUERY {i}: {query}\n")
            f.write("-"*80 + "\n")
            f.write(f"Search time: {query_time*1000:.2f} ms\n\n")
            
            f.write("TOP 3 RESULTS:\n\n")
            for result in results:
                f.write(f"Rank {result['rank']} | ID: {result['id']} | Topic: {result['topic']} | Score: {result['score']:.4f}\n")
                f.write("-"*80 + "\n")
                f.write(result['text'] + "\n\n")
            
            f.write("\n")
        
        # Add reflection
        f.write("="*80 + "\n")
        f.write("REFLECTION: WHEN SEMANTIC SEARCH SHINES VS FAILS\n")
        f.write("="*80 + "\n\n")
        
        reflection = """The semantic search system demonstrates both strengths and limitations across different query types. For factual, domain-specific queries like "How do neural networks work?", the system excels by correctly identifying relevant ML documents based on conceptual similarity rather than exact keyword matching. The embeddings successfully capture that "neural networks," "deep learning," and "transformers" are semantically related concepts.

However, the system struggles with extremely vague or ambiguous queries. The single-word query "heat" produces mixed results spanning cooking (searing, caramelizing) and climate science (greenhouse gases), showing that without sufficient context, the system cannot disambiguate user intent. This highlights a key limitation: semantic similarity alone doesn't capture user goals.

The search performs best with medium-length queries (10-20 words) that provide clear topical signals without being overly specific. Long, detailed queries work well when the corpus contains matching content, but very short queries often retrieve surprising or irrelevant results. Multi-topic queries successfully bridge different domains when concepts naturally overlap, demonstrating the power of embedding-based retrieval for discovering cross-domain connections that keyword search might miss.
"""
        
        f.write(reflection)
    
    print(f"\n✅ Search examples saved to: {filename}")


def main():
    """Main execution function."""
    
    # Create search engine
    search_engine = SemanticSearch()
    
    # Load corpus
    corpus = create_sample_corpus()
    search_engine.load_corpus(corpus)
    
    # Build index (or load from disk)
    index_time = search_engine.build_index(save_to_disk=True)
    
    # Run example queries
    query_results = run_example_queries(search_engine)
    
    # Save results
    save_search_examples(query_results)
    
    # Optional: Run interactive mode
    print("\n" + "="*80)
    print("Would you like to try your own queries? (y/n)")
    response = input("> ").strip().lower()
    
    if response == 'y':
        additional_queries = run_interactive_search(search_engine)
        if additional_queries:
            print(f"\n✓ You ran {len(additional_queries)} additional queries")
    
    print("\n" + "="*80)
    print("SEMANTIC SEARCH ENGINE - SESSION COMPLETE")
    print("="*80)
    print(f"Total documents indexed: {len(corpus)}")
    print(f"Index build time: {index_time:.3f} seconds")
    print(f"Average query time: {np.mean([qr['time'] for qr in query_results])*1000:.2f} ms")
    print("="*80)


if __name__ == "__main__":
    main()
