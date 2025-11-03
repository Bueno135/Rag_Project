# utils/search.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.services.cache import create_cache, get_cache, set_cache

# Garante que o cache existe
create_cache()

# 1️⃣ Pergunta do usuário
query = input("Digite sua pergunta: ")

# 2️⃣ Tenta buscar no cache
cached = get_cache(query)
if cached:
    print("\n⚡ Resultado do cache:")
    print(cached["text"])
    exit()  # encerra o programa aqui

# 3️⃣ Carrega o PDF
loader = PyPDFLoader("data/curriculo.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
texts = [chunk.page_content for chunk in chunks]

# 4️⃣ Gera embeddings
embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents(texts)
query_vector = embeddings.embed_query(query)

# 5️⃣ Calcula similaridades
cosine_scores = cosine_similarity([query_vector], vectors)[0]
dot_scores = np.dot(vectors, query_vector)

best_cosine_idx = int(np.argmax(cosine_scores))
best_dot_idx = int(np.argmax(dot_scores))

best_text = texts[best_cosine_idx][:600]

# 6️⃣ Exibe resultados
print("\n🔹 COSINE RESULT:")
print(texts[best_cosine_idx][:600], "...")
print("\n🔸 DOT-PRODUCT RESULT:")
print(texts[best_dot_idx][:600], "...")

# 7️⃣ Armazena no cache
set_cache(query, {"text": best_text})
