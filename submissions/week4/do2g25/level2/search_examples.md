# Semantic Search Engine - Query Examples and Analysis

## Query 1: "black holes and gravity"
**Category:** Factual, specific scientific query

### Results:
1. **[Doc 12]** Score: 0.3822
   > Knife skills are foundational in cooking. The proper grip, claw technique, and consistent cuts im...

2. **[Doc 28]** Score: 0.3495
   > Fiber optic cables transmit data as light pulses through glass fibers, enabling high-speed intern...

3. **[Doc 1]** Score: 0.3372
   > Mars has two small moons named Phobos and Deimos. Scientists believe they may be captured asteroi...

**Analysis:** Poor retrieval - failed to surface the black holes document (Doc 2). Retrieved unrelated cooking and technology docs, with only the third result being space-related.

---

## Query 2: "ancient knowledge and libraries"
**Category:** Vague, historical query with multiple potential matches

### Results:
1. **[Doc 15]** Score: 0.4403
   > The Silk Road was an ancient network of trade routes connecting East and West, facilitating not j...

2. **[Doc 24]** Score: 0.4257
   > Blockchain technology creates immutable distributed ledgers through cryptographic hashing and con...

3. **[Doc 8]** Score: 0.4148
   > Sourdough bread relies on wild yeast and bacteria for fermentation, creating a tangy flavor and c...

**Analysis:** Missed the Library of Alexandria document (Doc 14) entirely. Only the first result is historically relevant. Blockchain and sourdough are completely off-topic.

---

## Query 3: "how does heating sugar create sweet flavors and brown colors in cooking"
**Category:** Long, detailed multi-concept query

### Results:
1. **[Doc 12]** Score: 0.5140
   > Knife skills are foundational in cooking. The proper grip, claw technique, and consistent cuts im...

2. **[Doc 11]** Score: 0.4911
   > Sous vide cooking involves sealing food in bags and cooking in precisely controlled water baths, ...

3. **[Doc 4]** Score: 0.4644
   > Saturn's rings are made mostly of ice particles, with sizes ranging from tiny grains to house-siz...

**Analysis:** Complete failure - missed both caramelization (Doc 10) and Maillard reaction (Doc 7) documents. Retrieved generic cooking docs and an irrelevant space document.

---

## Query 4: "fermentation"
**Category:** Short, single-word query

### Results:
1. **[Doc 25]** Score: 0.2445
   > The transistor, invented in 1947, is the fundamental building block of modern electronics. Billio...

2. **[Doc 24]** Score: 0.1578
   > Blockchain technology creates immutable distributed ledgers through cryptographic hashing and con...

3. **[Doc 28]** Score: 0.1488
   > Fiber optic cables transmit data as light pulses through glass fibers, enabling high-speed intern...

**Analysis:** Catastrophic failure - retrieved only technology documents, completely missing the sourdough fermentation document (Doc 8). Very low scores indicate poor matching.

---

## Query 5: "computers and data transmission"
**Category:** Technology-focused, multi-concept query

### Results:
1. **[Doc 4]** Score: 0.4442
   > Saturn's rings are made mostly of ice particles, with sizes ranging from tiny grains to house-siz...

2. **[Doc 26]** Score: 0.4299
   > 5G networks use higher frequency bands and advanced antenna technology to achieve lower latency a...

3. **[Doc 7]** Score: 0.4210
   > Maillard reaction occurs when proteins and sugars in food are heated, creating complex flavors an...

**Analysis:** Partial success - Doc 26 (5G networks) is relevant for data transmission, but top result is about Saturn's rings and third is about cooking chemistry. Missed fiber optics (Doc 28) and machine learning (Doc 22).

---

## Performance Metrics

- **Corpus Indexing Time:** ~0.023s (23ms)
- **Average Query Time:** 0.35ms
- **Per-Document Indexing:** 0.78ms
- **Scores Range:** 0.15 - 0.51 (relatively low, indicating weak matches)

---

## Reflection: When Retrievals Shine vs. Fail

**Critical Findings:** This simple hash-based embedding approach fails dramatically on nearly all queries, demonstrating that lexical overlap alone is insufficient for semantic search. The system retrieved completely irrelevant documents for highly specific queries (knife skills for "black holes," Saturn's rings for "computers"), suggesting the hashing collisions and character-level features create spurious matches. Even when queries contained exact terminology present in documents (like "fermentation"), the system failed to retrieve them, indicating the embedding space doesn't preserve semantic relationships.

**Why It Fails:** The deterministic character/word hashing creates essentially random projections that don't capture meaning. Words like "ancient" and "blockchain" may hash to similar dimensions by chance, while semantically related terms like "black holes" and "gravity" may hash far apart. Without pre-trained language understanding, the system cannot recognize synonyms, related concepts, or domain-specific terminology.

**Path Forward:** This experiment powerfully demonstrates why production systems use transformer-based embeddings (sentence-transformers, OpenAI). These models learn semantic relationships from billions of text examples, enabling them to understand that "fermentation" relates to "sourdough" and "caramelization" relates to "heating sugar." The current implementation serves as a cautionary tale: semantic search requires semantically meaningful embeddings, not just mathematical similarity metrics.

---

## Notes

- The simple hash-based embedding proved inadequate for real semantic search
- Low similarity scores (< 0.52) across all queries indicate poor embedding quality
- Production systems MUST use pre-trained models like `sentence-transformers/all-MiniLM-L6-v2`
- This failure actually demonstrates the importance of proper embedding models
- Consider this a baseline showing what NOT to do in production
