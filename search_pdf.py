import chromadb
import ollama
from sentence_transformers import CrossEncoder

CHROMA_DB_PATH = 'dexter_db_pdf_bge'
COLLECTION_NAME = 'dexter_pdf_docs_bge'

print(f"Loading ChromaDB client (path: '{CHROMA_DB_PATH}')...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' loaded. Total chunks: {collection.count()}")
except Exception as e:
    print(f"!! Error loading collection: {e}")
    print("Please make sure you have run ingestion.py first.")
    exit()

reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Cross-Encoder model loaded.")


my_queries_list = [
    "Who are Dexter's main antagonists?",
    "Who are Dexter's complex relationships?"
]

my_query_for_llm = "Who are Dexter's main antagonists or complex relationships in the first three seasons?"
hyde_prompt = f"""
You are an expert on the TV show 'Dexter'.
Please write a short, hypothetical passage that answers the question below.
You can hallucinate or make up details if needed, but the answer must look like a real factual paragraph.
Do NOT include any introductory text like "Here is the answer". Output ONLY the passage.

Question: {my_query_for_llm}
"""

try:
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': hyde_prompt,
        }
    ])

    hyde_hallucination = response['message']['content']
    print(f"\n--- LLM ANSWER ---")
    print(hyde_hallucination)

except Exception as e:
    print(f"\n!! Error communicating with Ollama: {e}")

k_initial_results = 10

print(f"Query: '{[hyde_hallucination]}' (k={k_initial_results})")

# hyDE + MultyQuery reslts
ready_to_go = [hyde_hallucination] + my_queries_list
results = collection.query(
    query_texts=ready_to_go,
    n_results=k_initial_results,
)


all_flat_texts = []
all_flat_metadatas = []

for sublist_texts, sublist_metas in zip(results['documents'], results['metadatas']):
    for text, meta in zip(sublist_texts, sublist_metas):
        all_flat_texts.append(text)
        all_flat_metadatas.append(meta)

seen_texts = set()
unique_texts = []
unique_metadatas = []

for text, meta in zip(all_flat_texts, all_flat_metadatas):
    if text not in seen_texts:
        seen_texts.add(text)
        unique_texts.append(text)
        unique_metadatas.append(meta)


pairs_for_reranker = [ (my_query_for_llm, chunk) for chunk in unique_texts ]
new_scores = reranker_model.predict(pairs_for_reranker)

res = list(zip(new_scores, unique_texts, unique_metadatas))
res.sort(key=lambda x: x[0], reverse=True)
top_3_results = res[0:3]


context_for_ollama = []

for score, text, meta in top_3_results:
    print(f"\nResult (Score: {score:.4f}):")
    print(f"  Text: {text}")
    print(f"  Source: {meta['source_file']} (Page {meta['page_number']})")
    context_for_ollama.append(text)

context_string = "\n\n".join(context_for_ollama)

print(context_string)

prompt = f"Use ONLY this context to answer the question: \n\nContext: {context_string}\n\nQuestion: {my_query_for_llm}"

print("\n--- SENDING TO LLM ---")
print(f"Prompt: {prompt}")

try:
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': prompt,
        }
    ])

    ollama_answer = response['message']['content']
    print(f"\n--- LLM ANSWER ---")
    print(ollama_answer)

except Exception as e:
    print(f"\n!! Error communicating with Ollama: {e}")
    print("Make sure Ollama is running (e.g., with 'ollama serve' in the terminal).")