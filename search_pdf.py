import chromadb
import ollama
from sentence_transformers import CrossEncoder

CHROMA_DB_PATH = 'dexter_db_pdf'
COLLECTION_NAME = 'dexter_pdf_docs'


print(f"Loading ChromaDB client (path: '{CHROMA_DB_PATH}')...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' loaded. Total chunks: {collection.count()}")
except Exception as e:
    print(f"!! Error loading collection: {e}")
    print("Please make sure you have run ingestion.py first.")
    exit()

print("Loading Cross-Encoder model...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Cross-Encoder model loaded.")


my_query = "Who is The Ice Truck Killer?"
k_initial_results = 10

print(f"\n--- 🔎 STAGE 1: RETRIEVAL ---")
print(f"Query: '{my_query}' (k={k_initial_results})")

results = collection.query(
    query_texts=[my_query],
    n_results=k_initial_results,
    where={"source_file": "dexter_season1.pdf"}
)

context_texts = results['documents'][0]
metadatas = results['metadatas'][0]

print(f"Retrieved {len(context_texts)} initial candidates.")

print(f"\n--- STAGE 2: RERANKING ---")

pairs_for_reranker = [ (my_query, chunk) for chunk in context_texts ]
new_scores = reranker_model.predict(pairs_for_reranker)

res = list(zip(new_scores, context_texts, metadatas))
res.sort(key=lambda x: x[0], reverse=True)
top_3_results = res[0:3]

print("\n--- 🏆 STAGE 3: TOP 3 RERANKED RESULTS ---")

context_for_ollama = []

for score, text, meta in top_3_results:
    print(f"\nResult (Score: {score:.4f}):")
    print(f"  Text: {text}")
    print(f"  Source: {meta['source_file']} (Page {meta['page_number']})")
    context_for_ollama.append(text)

context_string = "\n\n".join(context_for_ollama)

print("\n--- 🧠 FINAL CONTEXT FOR LLM ---")
print(context_string)

prompt = f"Use ONLY this context to answer the question: \n\nContext: {context_string}\n\nQuestion: {my_query}"

print("\n--- 🤖 SENDING TO LLM ---")
print(f"Prompt: {prompt}")

try:
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': prompt,
        }
    ])

    ollama_answer = response['message']['content']
    print(f"\n--- ✅ LLM ANSWER ---")
    print(ollama_answer)

except Exception as e:
    print(f"\n!! Error communicating with Ollama: {e}")
    print("Make sure Ollama is running (e.g., with 'ollama serve' in the terminal).")