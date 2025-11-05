import chromadb
import os

client = chromadb.PersistentClient(path="dexter_db")
print("Запущено клієнт ChromaDB (збереження у папку 'dexter_db')")

collection = client.get_or_create_collection(name="dexter_docs")
print("Колекцію 'dexter_docs' завантажено/створено.")

RESUMES_DIR = 'dexter_data'
document_texts = []
document_metadatas = []
document_ids = []
doc_id_counter = 0

print(f"\nПочаток індексації з папки '{RESUMES_DIR}'...")

for filename in os.listdir(RESUMES_DIR):
    if filename.endswith('.txt'):
        filepath = os.path.join(RESUMES_DIR, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        document_texts.append(text)

        document_metadatas.append({"source_file": filename})

        document_ids.append(str(doc_id_counter))

        print(f"  - Знайдено ID {doc_id_counter} ({filename})")
        doc_id_counter += 1

collection.add(
    documents=document_texts,
    metadatas=document_metadatas,
    ids=document_ids
)

print(f"\nУспішно додано {len(document_ids)} документів у колекцію 'dexter_docs'.")
print("Chroma автоматично створила для них ембединги.")