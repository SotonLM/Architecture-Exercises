
import time
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "sentence-transformers is not installed.\n"
        "Run: uv pip install sentence-transformers"
    )


MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

#CORPUS - 20-30 paragraphs across topics
documents = [
    #technology
    "Python is a high-level programming language favored for its clear syntax and readability. It is widely used in web development, data science, and artificial intelligence.",
    "Machine learning models learn patterns from data to make predictions without explicit programming. They require large datasets and computational power for training.",
    "JavaScript is essential for web development, enabling interactive features on websites. It runs in browsers and has expanded to server-side development with Node.js.",
    "Cloud computing allows businesses to access computing resources over the internet on-demand. This includes storage, processing power, and software services.",
    "Artificial intelligence systems can perform tasks that typically require human intelligence. Applications include image recognition, natural language processing, and autonomous vehicles.",
    "Blockchain technology provides a decentralized and secure way to record transactions. It is the foundation for cryptocurrencies like Bitcoin and Ethereum.",

    #cooking    
    "Sourdough bread baking requires maintaining a live culture of wild yeast and bacteria. The fermentation process develops complex flavors and improves digestibility.",
    "Italian cooking emphasizes regional ingredients like olive oil, tomatoes, and fresh herbs. Dishes vary significantly from northern to southern Italy.",
    "French pastry techniques involve precise measurements and temperature control for perfect results. Key skills include laminating dough for croissants and tempering chocolate.",
    "Vegetarian cuisine focuses on plant-based proteins like beans, lentils, and tofu. Proper nutrition planning ensures adequate protein and essential nutrients.",
    "Knife skills are fundamental in professional kitchens for efficient and safe food preparation. Proper technique includes the claw grip and rocking motion.",
    "Fermentation preserves foods while enhancing flavor and nutritional value through microbial action. Examples include kimchi, sauerkraut, and kombucha.",

    #sports
    "Basketball strategy involves both offensive plays and defensive formations to outscore opponents. Team coordination and individual skills are equally important.",
    "Soccer tactics include formations like 4-4-2 and pressing strategies to regain possession. Set pieces like corners and free kicks often decide matches.",
    "Tennis requires excellent hand-eye coordination and strategic shot placement to win points. Different court surfaces affect ball speed and player movement.",
    "Marathon training involves building endurance through long runs and proper nutrition planning. Tapering before race day helps optimize performance.",
    "Swimming technique focuses on body position, breathing rhythm, and efficient stroke mechanics. Different strokes serve various purposes in competition.",
    "Weight training principles include progressive overload and proper form to build strength safely. Compound exercises work multiple muscle groups simultaneously.",

    #science
    "Quantum entanglement describes how particles remain connected regardless of distance between them. This phenomenon challenges classical physics concepts.",
    "Climate models predict future temperature changes based on greenhouse gas emission scenarios. These models help policymakers plan for environmental changes.",
    "Neuroscience research explores how brain structure relates to cognitive functions and behaviors. Advanced imaging techniques reveal neural connectivity patterns.",
    "Evolutionary biology explains how species adapt to environments through natural selection. Genetic variations provide the raw material for evolutionary changes.",
    "Renewable energy technologies like solar and wind power are crucial for reducing carbon emissions. Energy storage solutions address intermittency challenges.",
    "Genetic engineering techniques allow scientists to modify DNA for medical and agricultural applications. CRISPR technology enables precise gene editing."
]

def main() -> None:
    print("=== LEVEL 2: TINY SEMANTIC SEARCH ===")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    
    #log embeddings time
    start = time.perf_counter()
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    corpus_time = time.perf_counter() - start #added corpus time
    print(f"Embedded {len(documents)} documents in {corpus_time:.2f}s")

    while True:
        query = input("\nEnter search query: ") #Accept a query string via `input()`.

        start = time.perf_counter() 
        query_embedding = model.encode([query], normalize_embeddings=True) #embed the query
        similarities = query_embedding @ doc_embeddings.T #compute cosine similarity against corpus
        query_time = time.perf_counter() - start #log query time
        
        others = list(enumerate(similarities[0]))
        top_matches = sorted(others, key=lambda item: item[1], reverse=True)[:TOP_K] #return the top-`k` (default `k=3`)

        print(f"\nQuery processed in {query_time:.3f}s")
        
        #results along with document IDs/indices, snippets, and similarity scores.
        print(f"\nTop {TOP_K} results for: '{query}'")
        for rank, (doc_idx, score) in enumerate(top_matches, start=1): 
            print(f"  #{rank}  cosine={score:.3f}  →  Doc[{doc_idx}] {documents[doc_idx][:100]}...") 

if __name__ == "__main__":
    main()