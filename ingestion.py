import fitz
import os
import chromadb
from chromadb.utils import embedding_functions

embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")

PDF_DATA_DIR = 'dexter_pdf_data'
COLLECTION_NAME_BGE = 'dexter_pdf_docs_bge'
CHROMA_DB_PATH_BGE = 'dexter_db_pdf_bge'

client = chromadb.PersistentClient(path=CHROMA_DB_PATH_BGE)

try:
    client.delete_collection(name=COLLECTION_NAME_BGE)
    print(f"Old collection '{COLLECTION_NAME_BGE}' deleted.")
except chromadb.errors.NotFoundError:
    print(f"Collection '{COLLECTION_NAME_BGE}' not found. Creating a new one.")

collection = client.get_or_create_collection(
    name=COLLECTION_NAME_BGE,
    embedding_function=embedding_model,
    metadata={"hnsw:space": "cosine"}
)
print(f"Collection '{COLLECTION_NAME_BGE}' loaded/created with 'bge-base-en-v1.5' model.")


doc_id_counter = 0

for filename in os.listdir(PDF_DATA_DIR):
    if filename.endswith('.pdf'):
        filepath = os.path.join(PDF_DATA_DIR, filename)
        print(f"\n--- Processing file: {filename} ---")
        try:
            doc = fitz.open(filepath)

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()

                all_lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                LINES_PER_CHUNK = 5
                chunk_num_in_page = 0

                for i in range(0, len(all_lines), LINES_PER_CHUNK):
                    chunk_lines_list = all_lines[i: i + LINES_PER_CHUNK]
                    chunk_text = " ".join(chunk_lines_list)
                    chunk_num_in_page += 1
                    current_id = f"doc_{doc_id_counter}"

                    metadata = {
                        "source_file": filename,
                        "page_number": page_num + 1,
                        "chunk_number_in_page": chunk_num_in_page,
                        "first_line_in_group": i
                    }

                    collection.add(
                        documents=[chunk_text],
                        metadatas=[metadata],
                        ids=[current_id]
                    )
                    doc_id_counter += 1

            print(
                f"  > Found {chunk_num_in_page} chunks on page {page_num + 1} (grouping {LINES_PER_CHUNK} lines each).")
            doc.close()

        except Exception as e:
            print(f"!! Error processing file {filename}: {e}")

print(f"\n Ingestion complete. Total 'chunks' (documents) in database: {collection.count()}")