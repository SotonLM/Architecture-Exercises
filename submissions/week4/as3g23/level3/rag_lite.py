import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"

CORPUS = [
    "The Mars Rover has sent back new images of the red planet.",
    "NASA plans to send humans to the moon by 2025.",
    "SpaceX focuses on reusable rocket technology.",
    "Saturn's rings are made mostly of ice particles.",
    "The International Space Station orbits Earth every 90 minutes.",
    # THE MISSING LINK:
    "Astronauts eat freeze-dried food like shrimp cocktail and chocolate pudding.",
    "Pizza dough requires yeast, flour, water, and time to rise.",
    "Sushi is a traditional Japanese dish with vinegared rice.",
    "Python functions are defined using the def keyword.",
    "Java is a statically typed object-oriented language.",
    "React is a popular JavaScript library for building UIs."
]

print("Loading models... (this may take 10-20 seconds)")
embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
generator = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
corpus_embeddings = embedder.encode(CORPUS, normalize_embeddings=True)

def retrieve(query, k=2):
    query_emb = embedder.encode(query, normalize_embeddings=True)
    scores = corpus_embeddings @ query_emb
    top_indices = scores.argsort()[-k:][::-1]
    return [CORPUS[i] for i in top_indices]

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator.generate(**inputs, max_length=64, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("\n=== RAG-LITE SYSTEM ===")
    
    queries = [
        "What do astronauts eat?", 
        "What is Python?",
        "What is the capital of France?"
    ]

    for q in queries:
        print(f"\nUser Query: {q}")
        
        docs = retrieve(q)
        print(f"  [Retrieval]: Found {len(docs)} docs related to query.")

        context_text = "\n".join(docs)
        
        prompt = f"""Use the context below to answer the question. If the answer is not in the context, say "I don't know".

Context: {context_text}

Question: {q}
Answer:"""

        answer = generate_answer(prompt)
        print(f"  [RAG Answer]: {answer}")

if __name__ == "__main__":
    main()