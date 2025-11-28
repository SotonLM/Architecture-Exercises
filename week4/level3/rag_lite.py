import time

try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
except ImportError:
    raise SystemExit(
        "sentence-transformers is not installed.\n"
        "Run: uv pip install sentence-transformers"
    )

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

def clean_text(unclean_text):
    clean_text = " ".join(line.strip() for line in unclean_text)
    clean_text = clean_text.split("  ")
    # print(clean_text)
    return clean_text

def main() -> None:

    semantics = open("semantics.txt", "r", encoding="utf-8")
    text_lines = semantics.readlines()
    semantics.close()

    cleaned = clean_text(text_lines)

    generator = pipeline('text-generation', model='distilgpt2')

    model = SentenceTransformer(MODEL_NAME)

    start = time.perf_counter()
    corpus_embeddings = model.encode(cleaned, normalize_embeddings=True)

    print(f"Encoded {len(cleaned)} paragraphs in {time.perf_counter() - start:.2f}s")
    rag_comparison = open("rag_comparison.txt", "w", encoding="utf-8")

    for i in range(5):

        user_query = str(input("Enter query: "))
        rag_comparison.write(f"Query {i+1}: {user_query}\n")
        query_embedding = model.encode(user_query, normalize_embeddings=True)

        similarity = corpus_embeddings @ query_embedding
        sim_scores =[]
        for idx in range(len(cleaned)):
            sim_scores.append((idx, similarity[idx]))

        # get the topk results with their correspondiong paragraph
        top_matches = sorted(sim_scores, key=lambda item: item[1], reverse=True)[:TOP_K]

        # No context prompt
        no_con_prompt = f"Question: {user_query}"
        rag_comparison.write(f"No Context Prompt: {user_query}\n")
        no_con_output = generator(no_con_prompt, max_length=30)
        print("Prompt:",no_con_prompt)
        print(f"Answer: {no_con_output[0]['generated_text']}")
        rag_comparison.write(f"Answer: {no_con_output[0]['generated_text']}\n")
        print("-" * 50)

        # With Context prompt
        con_prompt = f"Context: {cleaned[top_matches[0][0]]}\n{cleaned[top_matches[1][0]]}\n{cleaned[top_matches[2][0]]}\nQuestion: {user_query}"
        rag_comparison.write(f"With Context Prompt: {user_query}\n")
        con_output = generator(con_prompt, max_length=30)
        print("Prompt:",con_prompt)
        print(f"Answer: {con_output[0]['generated_text']}")
        rag_comparison.write(f"Answer: {con_output[0]['generated_text']}\n")
        print("-" * 50)

    rag_comparison.close()

if __name__ == "__main__":
    main()

