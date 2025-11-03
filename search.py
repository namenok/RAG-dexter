import ollama
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading model and database...")

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

embeddings = np.load('embeddings.npy')

print(f"Database loaded successfully ({len(documents)} documents).")

def find_top_k(query_text, k=3):

    query_embedding = model.encode(query_text)

    scores = cosine_similarity([query_embedding], embeddings)[0]

    top_k_indices = np.argsort(scores)[:-k - 1:-1]

    results = []
    for idx in top_k_indices:
        results.append({
            'score': scores[idx],
            'text': documents[str(idx)]
        })

    return results


# to test
my_query = "Who is Dexter's sister?"

print(f"\nSearching for: '{my_query}'")
top_results = find_top_k(my_query, k=2)

print("\n--- Top Results ---")
for i, result in enumerate(top_results):
    print(f"Rank {i+1} (Score: {result['score']:.4f}):")
    print(result['text'])
    print("-------------------")

context_texts = [result['text'] for result in top_results]
my_glue = "\n\n"

context = my_glue.join(context_texts)
prompt = f'use only this contex: {context}. Give an answear on this question: {my_query}'

print("\n--- Sending to LLM ---")

response = ollama.chat(model='llama3', messages=[
    {
        'role': 'user',
        'content': prompt,
    }
])

print(response)
ollama_answer = str(response['message']['content'])
print(ollama_answer)