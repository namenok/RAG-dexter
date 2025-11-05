import chromadb
import ollama


CHROMA_DB_PATH = 'dexter_db_pdf'
COLLECTION_NAME = 'dexter_pdf_docs'

print(f"Запуск клієнта ChromaDB (база '{CHROMA_DB_PATH}')...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Колекцію '{COLLECTION_NAME}' завантажено. Загальна кількість чанків: {collection.count()}")
except Exception as e:
    print(f"!! Помилка завантаження колекції: {e}")
    print("Будь ласка, переконайся, що ти спершу запустив(ла) ingestion.py")
    exit()


my_query = "Who is The Ice Truck Killer?"
k_results = 3

print(f"\n--- 🔎 ЗВИЧАЙНИЙ ПОШУК ---")
print(f"Запит: '{my_query}' (k={k_results})")

results = collection.query(
    query_texts=[my_query],
    n_results=k_results,
    where={"source_file": "dexter_season1.pdf"}
)


context_texts = results['documents'][0]
metadatas = results['metadatas'][0]

for i in range(len(context_texts)):
    print(f"\nРезультат {i+1}:")
    print(f"  Текст: {context_texts[i]}")
    # А ось і користь від метаданих!
    print(f"  Джерело: {metadatas[i]['source_file']} (Стор. {metadatas[i]['page_number']})")