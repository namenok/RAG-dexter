import fitz
import os
import chromadb


PDF_DATA_DIR = 'dexter_pdf_data'
CHROMA_DB_PATH = 'dexter_db_pdf'
COLLECTION_NAME = 'dexter_pdf_docs'

print(f"Запуск клієнта ChromaDB (збереження у папку '{CHROMA_DB_PATH}')...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"Колекцію '{COLLECTION_NAME}' завантажено/створено.")


print(f"\nПочаток обробки файлів з '{PDF_DATA_DIR}'...")

doc_id_counter = 0


for filename in os.listdir(PDF_DATA_DIR):
    if filename.endswith('.pdf'):
        filepath = os.path.join(PDF_DATA_DIR, filename)

        print(f"\n--- Обробка файлу: {filename} ---")

        try:
            doc = fitz.open(filepath)


            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()


                chunks = [chunk.strip() for chunk in page_text.split('\n') if chunk.strip()]

                if not chunks:
                    print(f"  > Сторінка {page_num + 1} не містить тексту, пропускаємо.")
                    continue

                print(f"  > Знайдено {len(chunks)} чанків на сторінці {page_num + 1}.")


                chunk_num_in_page = 0
                for chunk_text in chunks:
                    chunk_num_in_page += 1


                    current_id = f"doc_{doc_id_counter}"

                    metadata = {
                        "source_file": filename,
                        "page_number": page_num + 1,
                        "chunk_number_in_page": chunk_num_in_page
                    }

                    collection.add(
                        documents=[chunk_text],
                        metadatas=[metadata],
                        ids=[current_id]
                    )

                    doc_id_counter += 1

            doc.close()

        except Exception as e:
            print(f"!! Помилка при обробці файлу {filename}: {e}")

print(f"\n🎉 Завантаження завершено. Загальна кількість 'чанків' (документів) у базі: {collection.count()}")