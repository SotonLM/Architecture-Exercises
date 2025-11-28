Subject: Joseph Arsany - Diagnostic Task Submission

LEVEL REACHED: [1/2/3/4]

LEVEL 1: ✅ 

Code Link: https://github.com/jrsany2903/Architecture-Exercises/blob/main/week4/level1/embeddings_basic.py
nearest_neighbors.txt : https://github.com/jrsany2903/Architecture-Exercises/blob/main/week4/level1/nearest_neighbours.txt



LEVEL 2: ✅ 
Code link: https://github.com/jrsany2903/Architecture-Exercises/blob/main/week4/level2/semantic_search.py
Corpus link: https://github.com/jrsany2903/Architecture-Exercises/blob/main/week4/level2/corpus.txt
Search sample link: https://github.com/jrsany2903/Architecture-Exercises/blob/main/week4/level2/search_examples.txt

Populated corpus.txt with the first section from wikipedia for Paris, sushi, einstein and the moon. I removed the [] refrences from the file because it was breaking the formatting.
Saved embeddings in corpus_embeddings.npy

<img src = "\level2results.png">

Search queries returned good results when the cosine similarity is >=0.65.

For the question about where is the eiffel tower, it correctly identified an association with paris, but given that the corpus does not contain a phrase like "the eiffel tower is in paris", the results feel very disjointed from the query.

LEVEL 3: ✅ [Option B]
Code link: https://github.com/jrsany2903/Architecture-Exercises/blob/main/week4/level3/rag_lite.py
Comparison text link: https://github.com/jrsany2903/Architecture-Exercises/blob/main/week4/level3/rag_comparison.txt

using ditilgpt2 results were slightly better but most answers were still lacking.
<img src = "\l3_1.png">
The context did not contain the answer and the transformer was not able to come up with it

<img src = "\l3_2.png">
The context did contain the answer but the transformer could not parse it correctly and could not form good sentences

<img src = "\l3_3.png">
The context had the answer and the transformer correctly parroted it but then contradicts itself later


LEVEL 4: ✅ [If completed]
- Project name: ___
- What it does: ___
- [Demo link]

TIME SPENT: ___ hours

WHAT I FOUND EASY: ___
WHAT I FOUND HARD: ___
QUESTIONS: ___


