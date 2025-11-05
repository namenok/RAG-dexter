import ollama
import chromadb

print("Loading ChromaDB client...")

client = chromadb.PersistentClient(path="dexter_db")

collection = client.get_collection(name="dexter_docs")

print(f"Database loaded successfully ({collection.count()} documents).")


my_query = "Who is Dexter's sister?"


print(f"\nSearching for: '{my_query}'")

results = collection.query(
    query_texts=[my_query],
    n_results=2
)

print("\n--- Top Results (from ChromaDB) ---")

context_texts = results['documents'][0]

scores = results['distances'][0]

for i, (text, score) in enumerate(zip(context_texts, scores)):

    print(f"Rank {i+1} (Score: {1.0 - score:.4f}):")
    print(text)
    print("-------------------")


my_glue = "\n\n"
context = my_glue.join(context_texts)
prompt = f'use only this contex: {context}. Give an answer on this question: {my_query}'

print("\n--- Sending to LLM ---")

response = ollama.chat(model='llama3', messages=[
    {
        'role': 'user',
        'content': prompt,
    }
])

ollama_answer = str(response['message']['content'])
print(f"\nLLM Answer:\n{ollama_answer}")