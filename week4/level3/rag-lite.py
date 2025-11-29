import os
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Retrieval and generation model settings
TRANSFORMER_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"
TOP_K = 5

# Read the corpus
def read_corpus(file_path: str) -> list[str]:
    corpus = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            formatted = line.strip().split(".")[0]
            if formatted:
                corpus.append(formatted)
    return corpus


# Save computed embeddings to disk for reuse
def store_embeddings(embeddings: np.ndarray, file_path: str) -> None:
    np.save(file_path, embeddings)


# Loads the cached embeddings or computes them if missing
def load_or_compute_embeddings(model: SentenceTransformer, corpus: list[str], path: str) -> np.ndarray:
    # Checks if the file exists
    if os.path.exists(path):
        print("Embeddings already stored.")
        return np.load(path)

    # Compute embeddings
    print("Computing embeddings for the corpus.")
    start = time.perf_counter()
    corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
    print(f"Encoded {len(corpus)} sentences in {time.perf_counter() - start:.2f}s")

    print("Storing embeddings.")
    store_embeddings(corpus_embeddings, path)
    return corpus_embeddings


# Retrieve top-k most similar corpus entries
def retrieve_top_k(corpus_embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    similarities = corpus_embeddings @ query_embedding
    top_indices = np.argsort(-similarities)[:top_k]
    top_scores = similarities[top_indices]
    return top_indices, top_scores


# Build context from retrieved documents
def build_context(corpus: list[str], indices: np.ndarray) -> str:
    lines = []
    for i, idx in enumerate(indices, start=1):
        lines.append(corpus[idx])
    return "\n".join(lines)


# Generate answer using the generator model
def generate_answer(model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, question: str, context: str | None = None, max_new_tokens: int = 128) -> str:
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = outputs = model.generate(input_ids=inputs.get("input_ids"), attention_mask=inputs.get("attention_mask"), max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# Main function
def main() -> None:
    print("=== LEVEL 3: RAG-LITE ===")

    print(f"Loading retriever: {TRANSFORMER_MODEL}")
    retriever = SentenceTransformer(TRANSFORMER_MODEL)

    # Get paths for corpus and embeddings
    corpus_path = "level2/corpus.txt"
    embeddings_path = "level2/embeddings.npy"

    # Loads corpus and its embeddings
    corpus = read_corpus(corpus_path)
    corpus_embeddings = load_or_compute_embeddings(retriever, corpus, embeddings_path)

    # loads generator model
    print(f"Loading generator: {GEN_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    generator = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

    queries = [
        "What is quantum computing and what problems can it solve?",
        "What is university like socially and academically?",
        "How is AI used in everyday life?",
        "What skills do I develop by studying AI at university?",
        "How fast is technology changing today?",
        "Why can university feel overwhelming at first?",
        "How does cloud computing affect AI systems?",
        "What is the role of ethics in modern technology?",
        "What makes campus life unique?",
        "How does studying at university shape your future career?",
    ]

    # Iterate over each query
    for i, query in enumerate(queries, start=1):
        print()
        # Embed query and retrieve top-k documents
        start = time.perf_counter()
        query_embedding = retriever.encode([query], normalize_embeddings=True)[0]
        indices, scores = retrieve_top_k(corpus_embeddings, query_embedding, TOP_K)
        elapsed = time.perf_counter() - start
        print(f"Retrieval completed in {elapsed:.2f}s")

        # Display retrieved documents
        print("Context:")
        for idx in indices:
            print(f"[DOC {idx}]: {corpus[idx]}")

        print()
        print(f"Query {i}: {query}")

        # Build context and generate answers
        context = build_context(corpus, indices)

        answer_no_context = generate_answer(generator, tokenizer, query, context=None)
        answer_with_context = generate_answer(generator, tokenizer, query, context=context)

        print()
        print("Answer WITHOUT context: ")
        print(f"    {answer_no_context}")

        print()
        print("Answer WITH context: ")
        print(f"    {answer_with_context}")

if __name__ == "__main__":
    main()