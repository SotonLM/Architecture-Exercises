"""
LEVEL 4: Hybrid Property Advisor System with Prompt Engineering
------------------------------------------------------------------------------------------------
NOTE: Includes sentence-transformers library, install using: uv pip install sentence-transformers
------------------------------------------------------------------------------------------------


The project purpose is to use data retrieval techniques to inform an answer based on a specific 
domain (as a Property Advisor in the Southampton area). It will use prompt engineering to classify
the initial input and from their choose an appropriate data retrieval technique (sql gen / embedding search)
to retrieve the relevant data. It will then use the retrieved data to answer the user's question.

This system demonstrates:
- Prompt-based query classification (routing)
- Structured data retrieval (SQL database with model-generated queries)
- Semantic search (vector embeddings)
- Answer synthesis using retrieved context

------------------------------------------------------------------------------------------------

STRUCTURE - (Process flow from input to output):

INPUT: User natural language query (e.g., "What's the average house price in Southampton?")

STEP 1: Query Classification (First Reprompting) - Classify query as FACTUAL or SEMANTIC to route to 
appropriate retrieval method. [Model: google/flan-t5-base]

STEP 2A: SQL Generation (Second Reprompting - FACTUAL path) - Convert natural language to SQL query,
execute against database, retrieve numerical data. [Model: google/flan-t5-base]

STEP 2B: Semantic Search (SEMANTIC path) - Generate query embeddings, compute cosine similarity with 
article embeddings, retrieve top-k similar articles. [Embedding: sentence-transformers/all-MiniLM-L6-v2]

STEP 3: Answer Synthesis (Third Reprompting) - Synthesize final answer from retrieved data/context using
original query and answer instructions. [Model: google/flan-t5-base]

OUTPUT: Final natural language answer to user's query
------------------------------------------------------------------------------------------------
"""

from transformers import pipeline, AutoModel, AutoTokenizer
import numpy as np
import torch
import sqlite3
import os

# ------------------------------------------------------------------------------------------------
# DATABASE SETUP
# ------------------------------------------------------------------------------------------------

db_file = 'property_data.db'

def init_property_database(db_path):
    """Create and populate SQLite database with property data"""
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE properties (
            area TEXT PRIMARY KEY,
            avg_price INTEGER,
            crime_rate REAL,
            school_rating REAL,
            population INTEGER,
            public_transport_score REAL,
            parks_count INTEGER
        )
    ''')
    
    # Insert data
    property_data = [
        ('Southampton', 350000, 45.2, 7.5, 253651, 8.2, 12),
        ('Portsmouth', 280000, 52.1, 7.2, 248440, 8.5, 8),
        ('Winchester', 550000, 28.5, 9.1, 45184, 6.8, 6),
        ('Bournemouth', 420000, 38.9, 7.8, 187503, 7.5, 15)
    ]
    
    cursor.executemany('''
        INSERT INTO properties (area, avg_price, crime_rate, school_rating, population, public_transport_score, parks_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', property_data)
    
    conn.commit()
    conn.close()
    print(f"✓ Property database created: {db_path}")

# ------------------------------------------------------------------------------------------------
# SAMPLE DATA
# ------------------------------------------------------------------------------------------------

# Sample news articles (semantic/contextual data)
news_articles = [
    {
        'area': 'Southampton',
        'title': 'Southampton Named Most Vibrant City',
        'content': 'Southampton has been recognized for its vibrant cultural scene and strong community spirit. The city offers excellent nightlife, diverse restaurants, and numerous festivals throughout the year. Residents praise the friendly atmosphere and growing tech industry.'
    },
    {
        'area': 'Southampton',
        'title': 'Safety Improvements in Southampton',
        'content': 'Recent police initiatives have led to improved safety in Southampton. Community watch programs are active, and new CCTV installations have reduced crime rates. The waterfront area is particularly safe, with families often seen enjoying evening walks.'
    },
    {
        'area': 'Portsmouth',
        'title': 'Portsmouth Maritime Heritage',
        'content': 'Portsmouth is known for its rich maritime history and naval connections. The city has a bustling port area with excellent shopping and dining. The seaside location provides beautiful views, though the city center can get busy with tourists during summer.'
    },
    {
        'area': 'Portsmouth',
        'title': 'Portsmouth Transport Hub',
        'content': 'Portsmouth benefits from excellent public transport connections, making it easy to commute. The city has good train links to London and the surrounding areas. Traffic can be heavy in rush hours, but the ferry connections add to the city\'s appeal.'
    },
    {
        'area': 'Winchester',
        'title': 'Winchester: Historic Cathedral City',
        'content': 'Winchester offers a tranquil, historic atmosphere with stunning architecture. The city is known for its excellent schools and family-friendly environment. It has a slower pace of life compared to larger cities, with charming independent shops and cafes.'
    },
    {
        'area': 'Winchester',
        'title': 'Winchester Quality of Life',
        'content': 'Winchester consistently ranks high for quality of life. The city combines historic charm with modern amenities. Green spaces are plentiful, and crime rates are notably low. The downside is higher property prices, but residents value the peaceful lifestyle.'
    },
    {
        'area': 'Bournemouth',
        'title': 'Bournemouth Beach Life',
        'content': 'Bournemouth is famous for its beautiful beaches and seaside lifestyle. The town has a relaxed, holiday atmosphere with excellent coastal walks and water sports. The nightlife is vibrant, especially during summer months when tourists arrive.'
    },
    {
        'area': 'Bournemouth',
        'title': 'Bournemouth Retiree Appeal',
        'content': 'Bournemouth attracts many retirees due to its pleasant climate and slower pace. The town has good healthcare facilities and numerous parks. Property prices have risen, but the coastal location and quality of life make it appealing for families too.'
    }
]

# ------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------------------------

def cosine_sim(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, tokenizer, model, use_sentence_transformers=False):
    """Get embedding for text using the model's tokenizer and encoder"""
    if use_sentence_transformers:
        # sentence-transformers model - directly encode
        return model.encode(text, convert_to_numpy=True)
    else:
        # Transformers model - manual encoding
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embeddings

# ------------------------------------------------------------------------------------------------
# PROMPT ENGINEERING FUNCTIONS
# ------------------------------------------------------------------------------------------------

def classify_query(query, generator):
    """Use prompt engineering to classify if query is FACTUAL or SEMANTIC"""
    classification_prompt = f"""Classify the following real estate question as either FACTUAL or SEMANTIC.

FACTUAL questions ask for specific numerical data (prices, rates, counts, scores).
SEMANTIC questions ask about experiences, opinions, atmosphere, or qualitative descriptions.

Question: "{query}"

Classification (respond with only FACTUAL or SEMANTIC):"""

    output = generator(
        classification_prompt,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=False
    )
    
    # text2text-generation returns just the generated text (no prompt included)
    response = output[0]['generated_text'].strip().upper()
    
    if 'FACTUAL' in response:
        return 'FACTUAL'
    elif 'SEMANTIC' in response:
        return 'SEMANTIC'
    else:
        # Default heuristic if model output is unclear
        factual_keywords = ['price', 'cost', 'rate', 'score', 'count', 'number', 'how much', 'average', 'statistics']
        if any(keyword in query.lower() for keyword in factual_keywords):
            return 'FACTUAL'
        return 'SEMANTIC'

def query_factual_db(query, generator, db_path=db_file):
    """Query the SQLite database - uses model to generate SQL query"""
    sql_generation_prompt = f"""Generate SQL query. Format: SELECT column FROM properties WHERE area = 'CityName'

Table name: properties
Columns: avg_price, crime_rate, school_rating, population, public_transport_score, parks_count

Question to column mapping:
- price/cost → avg_price
- crime/safety → crime_rate  
- school/education → school_rating
- population/people → population
- transport → public_transport_score
- parks → parks_count

Examples:
Q: "What's the average house price in Southampton?"
A: SELECT avg_price FROM properties WHERE area = 'Southampton'

Q: "How many parks are in Bournemouth?"
A: SELECT parks_count FROM properties WHERE area = 'Bournemouth'

Q: "What's the crime rate in Winchester?"
A: SELECT crime_rate FROM properties WHERE area = 'Winchester'

Q: "{query}"
A:"""

    output = generator(
        sql_generation_prompt,
        max_new_tokens=30,
        temperature=0.0,
        do_sample=False
    )
    
    sql_query = output[0]['generated_text'].strip()
    print(f"Generated SQL: {sql_query}")
    
    # Extract SQL from response - look for SELECT statement
    if 'SELECT' in sql_query.upper():
        # Find the SELECT and extract from there
        start_idx = sql_query.upper().find('SELECT')
        sql_query = sql_query[start_idx:]
        # Remove any trailing text after the query
        # Look for common endings
        for end_marker in ['\n', ';', '```']:
            if end_marker in sql_query:
                sql_query = sql_query[:sql_query.find(end_marker)]
        sql_query = sql_query.strip()
    
    # Clean up any markdown formatting
    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
    
    # Post-process to fix common issues
    sql_query = sql_query.replace('park_count', 'parks_count')
    sql_query = sql_query.replace('CityName', '')  # remove placeholder
    sql_query = ' '.join(sql_query.split())  # Remove extra spaces
    
    print(f"Final SQL: {sql_query}")
    
    # Execute the SQL query
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()
        
        # Format results
        if not results:
            return "No data found for this query.", None
        
        # Build formatted response
        facts_list = []
        area = None
        
        for row in results:
            row_dict = dict(zip(column_names, row))
            if 'area' in row_dict:
                area = row_dict['area']
            
            for col, val in row_dict.items():
                if col == 'area':
                    continue
                # Format based on column type
                if col == 'avg_price':
                    facts_list.append(f"Average house price: £{val:,}")
                elif col == 'crime_rate':
                    facts_list.append(f"Crime rate: {val} per 1000 people")
                elif col == 'school_rating':
                    facts_list.append(f"School rating: {val}/10")
                elif col == 'population':
                    facts_list.append(f"Population: {val:,}")
                elif col == 'public_transport_score':
                    facts_list.append(f"Public transport score: {val}/10")
                elif col == 'parks_count':
                    facts_list.append(f"Number of parks: {val}")
                else:
                    facts_list.append(f"{col.replace('_', ' ').title()}: {val}")
        
        facts_text = f"Area: {area}\n" + "\n".join(facts_list) if area else "\n".join(facts_list)
        return facts_text, area
        
    except sqlite3.Error as e:
        # Fallback: try to extract area and use keyword matching
        print(f"SQL Error: {e}")
        print("Falling back to keyword-based retrieval")
        
        # Extract area from query
        areas = ['Southampton', 'Portsmouth', 'Winchester', 'Bournemouth']
        area = None
        for a in areas:
            if a.lower() in query.lower():
                area = a
                break
        
        if not area:
            return "Area not found in database.", None
        
        # Simple keyword-based field selection
        query_lower = query.lower()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        fields_to_get = []
        if 'price' in query_lower or 'cost' in query_lower:
            fields_to_get.append('avg_price')
        if 'crime' in query_lower or 'safe' in query_lower:
            fields_to_get.append('crime_rate')
        if 'school' in query_lower or 'education' in query_lower:
            fields_to_get.append('school_rating')
        if 'population' in query_lower or 'people' in query_lower:
            fields_to_get.append('population')
        if 'transport' in query_lower or 'commute' in query_lower:
            fields_to_get.append('public_transport_score')
        if 'park' in query_lower or 'green' in query_lower:
            fields_to_get.append('parks_count')
        
        if not fields_to_get:
            fields_to_get = ['avg_price', 'crime_rate', 'school_rating', 'population', 'public_transport_score', 'parks_count']
        
        fields_str = ', '.join(fields_to_get)
        fallback_query = f"SELECT {fields_str} FROM properties WHERE area = ?"
        cursor.execute(fallback_query, (area,))
        results = cursor.fetchone()
        conn.close()
        
        if results:
            facts_list = []
            field_names = {
                'avg_price': 'Average house price',
                'crime_rate': 'Crime rate',
                'school_rating': 'School rating',
                'population': 'Population',
                'public_transport_score': 'Public transport score',
                'parks_count': 'Number of parks'
            }
            
            for i, field in enumerate(fields_to_get):
                val = results[i]
                if field == 'avg_price':
                    facts_list.append(f"{field_names[field]}: £{val:,}")
                elif field == 'crime_rate':
                    facts_list.append(f"{field_names[field]}: {val} per 1000 people")
                elif field == 'school_rating':
                    facts_list.append(f"{field_names[field]}: {val}/10")
                elif field == 'population':
                    facts_list.append(f"{field_names[field]}: {val:,}")
                elif field == 'public_transport_score':
                    facts_list.append(f"{field_names[field]}: {val}/10")
                elif field == 'parks_count':
                    facts_list.append(f"{field_names[field]}: {val}")
            
            facts_text = f"Area: {area}\n" + "\n".join(facts_list)
            return facts_text, area
        
        return "No data found.", None

def query_semantic_db(query, tokenizer, model, top_k=3, use_sentence_transformers=False):
    """Query the vector embeddings database using semantic similarity"""
    # Get query embedding (tokenizer can be None if using sentence-transformers)
    query_embedding = get_embedding(query, tokenizer, model, use_sentence_transformers)
    
    # Calculate similarities with all articles
    similarities = []
    for article in news_articles:
        article_text = f"{article['title']}. {article['content']}"
        article_embedding = get_embedding(article_text, tokenizer, model, use_sentence_transformers)
        similarity = cosine_sim(query_embedding, article_embedding)
        similarities.append((similarity, article))
    
    # Sort by similarity and get top k
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_articles = similarities[:top_k]
    
    # Format context
    context_parts = []
    for sim, article in top_articles:
        context_parts.append(f"[{article['area']}] {article['title']}: {article['content']}")
    
    return "\n\n".join(context_parts)

def answer_question(query, facts_or_context, generator, query_type):
    """Use prompt engineering to synthesize an answer from retrieved information"""
    if query_type == 'FACTUAL':
        answer_prompt = f"""Given the following factual data, answer the user's question concisely and clearly.

Data:
{facts_or_context}

Question: "{query}"

Answer:"""
    else:  # SEMANTIC
        answer_prompt = f"""Given the following context from news articles about the area, answer the user's question in a natural, conversational way.

Context:
{facts_or_context}

Question: "{query}"

Answer:"""
    
    output = generator(
        answer_prompt,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    
    # text2text-generation returns just the generated text (no prompt included)
    answer = output[0]['generated_text'].strip()
    return answer

def process_query(query, generator, tokenizer, embedding_model, use_sentence_transformers=False):
    """Main function to process a query through the hybrid system"""
    print(f"\n{'='*80}")
    print(f"USER QUERY: {query}")
    print(f"{'='*80}")
    
    # Step 1: Classify query
    query_type = classify_query(query, generator)
    print(f"Classification: {query_type}")
    
    # Step 2: Retrieve information
    if query_type == 'FACTUAL':
        print("\n[Routing to Numerical Database]")
        facts_text, area = query_factual_db(query, generator)
        print(f"Retrieved Data:\n{facts_text}")
        retrieved_info = facts_text
    else:  # SEMANTIC
        print("\n[Routing to Vector Embeddings Database]")
        context_text = query_semantic_db(query, tokenizer, embedding_model, use_sentence_transformers=use_sentence_transformers)
        print(f"Retrieved Context:\n{context_text[:500]}...")
        retrieved_info = context_text
    
    # Step 3: Generate answer
    print("\n[Generating Answer]")
    answer = answer_question(query, retrieved_info, generator, query_type)
    print(f"\nANSWER: {answer}")
    
    return answer, query_type

# ------------------------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== LEVEL 4: HYBRID PROPERTY ADVISOR SYSTEM ===")
    
    # Initialize database
    init_property_database(db_file)
    
    # Initialize models
    print("\nInitializing models...")
    
    # Use FLAN-T5-base for better instruction following and SQL generation
    print("Loading text generation model (FLAN-T5-base)...")
    generator = pipeline('text2text-generation', model='google/flan-t5-base')
    
    # Try to use sentence-transformers for embeddings (best option)
    print("Loading embedding model...")
    use_sentence_transformers = False
    embedding_tokenizer = None
    embedding_model = None
    
    try:
        from sentence_transformers import SentenceTransformer
        print("Using sentence-transformers (all-MiniLM-L6-v2) for embeddings...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        use_sentence_transformers = True
        print("✓ sentence-transformers loaded successfully")
    except ImportError:
        print("sentence-transformers not available, using distilbert-base-uncased instead...")
        embedding_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        embedding_model = AutoModel.from_pretrained('distilbert-base-uncased')
        print("✓ distilbert-base-uncased loaded successfully")
    
    print(f"\nUsing local model: google/flan-t5-base")
    
    # Test queries
    test_queries = [
        "What's the average house price in Southampton?",
        "How safe is Southampton? Tell me about the crime rate.",
        "What's the vibe like in Winchester? Is it a good place to live?",
        "How many parks are in Bournemouth?",
        "What is life like in Portsmouth?",
    ]
    
    print("\n" + "="*80)
    print("TESTING HYBRID PROPERTY ADVISOR SYSTEM")
    print("="*80)
    
    for query in test_queries:
        try:
            answer, q_type = process_query(query, generator, embedding_tokenizer, embedding_model, use_sentence_transformers=use_sentence_transformers)
            print("\n")
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)

