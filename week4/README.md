# Week 4: Embeddings & Retrieval

Week 4 turns text-generation toys into usable systems. You will learn how to represent text as vectors, retrieve relevant context, and (optionally) inject that context into a lightweight generator for a RAG-lite experience.

**If you’re overwhelmed: finish Levels 1–2 and stop. That is enough for Week 4. Levels 3–4 are optional Tier 3 stretch goals.**

## Structure

```text
week4/
├── README.md                 # This file
├── diagnostic_task.py        # Embeddings + retrieval starter
├── SETUP.MD *                # (Optional) reuse week3 setup if needed
├── pyproject.toml *          # Bring in dependencies if you isolate week4
├── submission_format.txt *   # Copy if running week4 standalone
├── level1/
│   └── README.md             # Embeddings warm-up (Tier 1)
├── level2/
│   └── README.md             # Semantic search system (Tier 2)
├── level3/
│   └── README.md             # Quality metric or RAG-lite (Tier 3)
└── level4/
    └── README.md             # Freestyle retrieval-based build (Stretch)
```

> \* Week 4 reuses the Week 3 `uv` environment. Only add new files if you want a standalone copy.

## Why This Week

- You already know how to install models, tune generation, and reason about quality (Weeks 1–3).
- The missing piece is *control*: using embeddings + retrieval to ground your generations.
- These primitives (vectors, similarity search, retrieval-augmented generation) power almost every “real” LLM application.

## Level Overview & Tiers

| Level | Theme | Tier Mapping |
| --- | --- | --- |
| Level 1 | Compute embeddings + nearest neighbours | Tier 1 (required baseline) |
| Level 2 | Build a tiny semantic search engine | Tier 2 (core project) |
| Level 3 | Pick **one**: retrieval metric **or** RAG-lite | Tier 3 (advanced) |
| Level 4 | Freestyle retrieval-based tool | Tier 3+ stretch / portfolio |

> **Plain-language RAG-lite:** retrieve the top-k relevant paragraphs, paste them above the question, and feed the combined text to your generator. That’s it.

## Quick Start

1. **Reuse the Week 3 environment** (or create a new one with `uv venv`).
2. `uv pip install sentence-transformers` (pulls in transformers, torch, numpy). Add extra packages only if you need them.
3. Work through each level in order; submit artefacts as described in the level READMEs.

## Deliverables

- Level-specific code files (see level READMEs for expected filenames).
- Text artefacts (e.g., `nearest_neighbours.txt`, `search_examples.txt`, `quality_results.txt`, `rag_comparison.txt`).
- Level 4 submissions must include a short description, sample outputs, and a "What I'd improve next" note.

## Submission & Due Date

- Follow the same submission process as Week 3 (Teams #architecture channel, Wednesday 11:59 PM unless told otherwise).
- Use the Week 4 section of the standard submission template.

## Resources

- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Feature Extraction Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.FeatureExtractionPipeline)
- [Cosine Similarity (sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- [RAG Overview](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## Tips

- Start with a *tiny* corpus so you can iterate quickly.
- Cache embeddings to disk once Level 2 works; it will save you time for Level 3/4.
- Quality thinking matters: log timings, note retrieval failures, and capture before/after examples.

