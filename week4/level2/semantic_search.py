import time
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

corpus = [
    "Apple grows on trees in orchards around the world. Apples usually grow in climates that have cold winters and moderate summer temperatures. Apples come in different colours like red, green and yellow.",
    "Apple pies are desert made with apples and cinnamon. They are sometimes served with ice cream or whipped cream. Peaple usually eat them during autumn or winter.",
    "Apples are a type of fruit that grow on apple trees. They are usually round and can be red, green or yellow. Apples are sweet and crunchy, and people often eat them as a snack or use them in cooking.",
    "Apples can be used to make many delicious foods. Some popular apple dishes include apple pie, apple crumble, and apple sauce. Apples can also be juiced or made into cider.",
    "Apples are healthy and nutritious. They are a good source of fiber and vitamin C. Eating apples can help improve digestion and boost the immune system.",

    "Ice cream is cold so it's good for the summer. There are lots of flavors like chocolate, vanilla and strawberry. Ice cream can come in a cone or a cup.",
    "Summer is hot so people like to eat ice cream. People usually dont eat ice cream during the winter because it's cold. People can buy ice creams from supermarkets or ice cream shops.",
    "ice cream is a sweet treat made from dairy products. It is usually served frozen and comes in many different flavors. People often enjoy ice cream on hot days or as a dessert after meals.",
    "Ice cream can be made at home using simple ingredients like milk, sugar and flavorings. There are also many different types of ice cream, such as gelato, sorbet and frozen yogurt.",
    "Ice cream is a popular dessert around the world. It is often enjoyed at parties, festivals and other celebrations. Many people have their favorite ice cream flavors and toppings.",

    "Marathon runners eat lots of pasta. Pasta is a good source of carbohydrates which gives them energy. They also eat other foods to stay healthy and strong.",
    "I go for a jog every Wednesday. Jogging is a good way to stay fit and healthy. I usually jog in gym on campus.",
    "running is a popular form of exercise that many people enjoy. It helps to improve cardiovascular health, build muscle strength and boost mental well-being.",
    "Jogging can be done outdoors or on a treadmill. Many people like to jog in parks or along scenic routes. It is important to wear comfortable shoes and stay hydrated while jogging.",
    "Jogging regularly can help to improve endurance and overall fitness levels. It is also a great way to relieve stress and clear the mind.",

    "Museums display ancient or cool things. People visit museums to learn about history, art and culture. Some museums have famous paintings or artifacts.",
    "Famous museums have lots of visitors every year. People come from all over the world to see the exhibits. Museums often have special events or tours for visitors.",
    "Museums are institutions that collect, preserve and display objects of historical, cultural or scientific significance. They provide educational opportunities for people of all ages and backgrounds.",
    "Museums can focus on a variety of topics, such as art, history, science or natural history. Many museums also offer interactive exhibits and hands-on activities for visitors.",
    "Visiting museums can be a fun and enriching experience. They provide a chance to learn about different cultures, time periods and scientific discoveries."
]

def main() -> None:
    print("=== LEVEL 2: TINY SEMANTIC SEARCH ===")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # embed corpus 
    print("\nembedding corpus...")
    start = time.perf_counter()
    corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
    print(f"Time taken: {time.perf_counter() - start:.2f}s")

    while True:
        query = input("\nEnter search query (or type 'exit'): ")
        if query.lower() == "exit":
            break

        # embed query + compute similarity
        start = time.perf_counter()
        query_embedding = model.encode([query], normalize_embeddings=True)
        scores = (corpus_embeddings @ query_embedding.T).squeeze()
        query_time = time.perf_counter() - start

        # top K results
        top_indices = scores.argsort()[::-1][:TOP_K]

        print(f"\ntime taken: {query_time:.3f}s")
        print("top 3 results:")
        for rank, idx in enumerate(top_indices, start=1):
            print(f"\nRank {rank}  similarity score: {scores[idx]:.3f}")
            print(f"ID [{idx}]: {corpus[idx][:100]}...")


if __name__ == "__main__":
    main()