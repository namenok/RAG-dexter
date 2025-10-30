import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading 'all-MiniLM-L6-v2' model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

RESUMES_DIR = 'dexter_data'
OUTPUT_EMBEDDINGS = 'embeddings.npy'
OUTPUT_DOCUMENTS = 'documents.json'

document_texts = []
document_ids = []

print(f"Starting indexing process from '{RESUMES_DIR}' directory...")
doc_id = 0
for filename in os.listdir(RESUMES_DIR):
    if filename.endswith('.txt'):
        filepath = os.path.join(RESUMES_DIR, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        document_texts.append(text)
        document_ids.append(doc_id)

        print(f"  - Found document ID {doc_id} ({filename})")
        doc_id += 1

print("\nCreating embeddings...")
embeddings = model.encode(document_texts, show_progress_bar=True)
print(f"Created {len(embeddings)} embeddings.")
print(f"Vector dimension: {embeddings[0].shape}")

print("\nSaving results...")

np.save(OUTPUT_EMBEDDINGS, embeddings)
print(f"  - Embeddings saved to {OUTPUT_EMBEDDINGS}")

documents_data = {str(id): text for id, text in zip(document_ids, document_texts)}

with open(OUTPUT_DOCUMENTS, 'w', encoding='utf-8') as f:
    json.dump(documents_data, f, ensure_ascii=False, indent=4)
print(f"  - Document texts saved to {OUTPUT_DOCUMENTS}")

print("\n Indexing complete! Our manual vector DB is ready.")