Subject: James Martineau - Diagnostic Task Submission week 4

LEVEL REACHED: [2]

LEVEL 1: ✅ [\[Code link\]](<level 1/embeddings_basic.py>) [\[Results link\]](<level 1/nearest_neighbours.txt>)

LEVEL 2: ✅ [\[Code link\]](<level 2/semantic_search.py>) [\[Results link\]](<level 2/search_examples.txt>) [\[Corpus link\]](<level 2/corpus.json>)
    - Retrievals do not do well accross multi-topic queries as they seem to focus on one topic. This is probably because all encodings will have similariety scores that are close together and I'm only retrieving the top 3 results.
    - Retrievals do well when the query contains words that are in the paragraphs or relate to a topic in the corpus but obvously do not perform well when the query doesnt contain a topic not covered in the corpus. 
    - Corpus needs to be extended a lot more with a lot more topics.

TIME SPENT: 2 hours

WHAT I FOUND EASY: Computing the similarity between each paragraph in the corpus and the provided query
WHAT I FOUND HARD: Constructing a diverse corpus
QUESTIONS: How can these embeddings be used to make the output of a generative model more coherent?