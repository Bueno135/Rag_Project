import sqlite3
import os
import json
import hashlib

# 🔹 Define caminho absoluto pro banco (independente de onde rodar o app)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "db")
os.makedirs(DB_DIR, exist_ok=True)  # garante que a pasta existe
DB_PATH = os.path.join(DB_DIR, "cache.db")

def create_cache():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            query_hash TEXT PRIMARY KEY,
            result_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_cache(query):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    cur.execute("SELECT result_json FROM cache WHERE query_hash=?", (query_hash,))
    row = cur.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def set_cache(query, result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    result_json = json.dumps(result)
    cur.execute("INSERT OR REPLACE INTO cache (query_hash, result_json) VALUES (?, ?)",
                (query_hash, result_json))
    conn.commit()
    conn.close()
