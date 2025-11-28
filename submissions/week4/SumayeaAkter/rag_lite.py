# week4/level3/rag_lite.py

import json
import time
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import pipeline


MODEL_NAME_EMB = "sentence-transformers/paraphrase-MiniLM-L6-v2"
MODEL_NAME_GEN = "google/flan-t5-small"
TOP_K = 3
OUTPUT_FILE = "week4/level3/rag_comparison.txt"


def load_corpus(path="week4/level2/corpus.json") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def embed_corpus(model, corpus):
    start = time.perf_counter()
    emb = model.encode(
        corpus,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    elapsed = time.perf_counter() - start
    print(f"Corpus embedded in {elapsed:.3f} seconds.")
    return emb

def search_top_k(query, model, corpus_emb, corpus):
    q_emb = model.encode([query], normalize_embeddings=True)
    sims = q_emb @ corpus_emb.T
    sims = sims.flatten()

    top_idx = sims.argsort()[::-1][:TOP_K]

    return [(idx, corpus[idx], float(sims[idx])) for idx in top_idx]


def build_prompt_no_context(query: str) -> str:
    return f"Question: {query}\nAnswer:"


def build_prompt_with_context(query: str, docs: List[str]) -> str:
    ctx = "\n".join([f"[DOC {i+1}] {d}" for i, d in enumerate(docs)])
    return f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"



def main():

    # 1) Load corpus
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} documents.")

    # 2) Load embedding model
    emb_model = SentenceTransformer(MODEL_NAME_EMB)
    corpus_emb = embed_corpus(emb_model, corpus)

    # 3) Load generator model
    print("\nLoading generation model...")
    generator = pipeline("text2text-generation", model=MODEL_NAME_GEN)
    print("Generation model loaded.")

    # 4) Ask user for queries
    print("\nEnter queries (type 'quit' to finish and generate rag_comparison.txt):\n")

    results_log = []

    while True:
        query = input("> ")

        if query.lower().strip() in {"quit", "exit"}:
            break

        # Retrieve docs
        top_docs = search_top_k(query, emb_model, corpus_emb, corpus)
        doc_texts = [doc for _, doc, _ in top_docs]

        # Build prompts
        prompt_no_ctx = build_prompt_no_context(query)
        prompt_ctx = build_prompt_with_context(query, doc_texts)

        # Generate both answers
        ans_no_ctx = generator(prompt_no_ctx, max_length=120)[0]["generated_text"]
        ans_ctx = generator(prompt_ctx, max_length=160)[0]["generated_text"]

        # Store for file writing later
        results_log.append({
            "query": query,
            "docs": top_docs,
            "no_context": ans_no_ctx,
            "with_context": ans_ctx
        })

        # Print immediate preview
        print("\n Baseline Answer ")
        print(ans_no_ctx)
        print("\n RAG-Lite Answer ")
        print(ans_ctx)
        print()

    # 5) Write everything to rag_comparison.txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("LEVEL 3 — RAG-LITE COMPARISON \n\n")

        for item in results_log:
            f.write(f"Query: {item['query']}\n\n")

            f.write("Retrieved Docs:\n")
            for idx, doc, score in item["docs"]:
                f.write(f"  Doc {idx} (score={score:.4f}):\n")
                f.write(f"    {doc}\n")
            f.write("\n")

            f.write("Answer WITHOUT context \n")
            f.write(item["no_context"] + "\n\n")

            f.write("Answer WITH context (RAG-Lite) \n")
            f.write(item["with_context"] + "\n\n")

            f.write("Your judgment: \n\n")
            f.write("-" * 50 + "\n\n")

        f.write("\nReflection\n")
        f.write("Write 1–2 paragraphs here comparing answer quality.\n")

    print(f"\nSaved results to: {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
