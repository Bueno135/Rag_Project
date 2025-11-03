from collections import defaultdict

# dicionário: session_id -> lista de mensagens [{role: "user"/"assistant", "text": "..."}]
_conversations = defaultdict(list)

def add_message(session_id: str, role: str, text: str):
    _conversations[session_id].append({"role": role, "text": text})

def get_history(session_id: str, limit: int = 10):
    # pega só as últimas N trocas
    return _conversations[session_id][-limit:]

def clear_history(session_id: str):
    _conversations[session_id] = []
