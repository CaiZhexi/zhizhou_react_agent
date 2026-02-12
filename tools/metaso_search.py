# tools/metaso_search.py
import requests
from config import METASO_API_KEY

def metaso_search(query: str) -> dict:
    url = "https://metaso.cn/api/v1/search"
    headers = {
        "Authorization": f"Bearer {METASO_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "scope": "webpage",
        "includeSummary": True,
        "size": "5",
        "includeRawContent": False,
        "conciseSnippet": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()  # ✅ 这就是“原始返回”
