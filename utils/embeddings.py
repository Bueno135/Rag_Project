# utils/embeddings.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()  # carrega do .env

# Carrega o PDF
loader = PyPDFLoader("data/curriculo.pdf")
docs = loader.load()

# Divide em chunks de até 1000 caracteres com sobreposição de 200
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Gera os embeddings
embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])

print(f"{len(vectors)} embeddings gerados com sucesso!")
