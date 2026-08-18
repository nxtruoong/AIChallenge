import os
import re
import json
import requests
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class DistilledQueryPayload:
    visual_subqueries: list[str] = field(default_factory=list)
    bge_dense_query: str = ""
    english_keywords: list[str] = field(default_factory=list)
    vietnamese_keywords: list[str] = field(default_factory=list)

    @property
    def visual_query(self) -> str:
        return " ".join(self.visual_subqueries) if self.visual_subqueries else self.bge_dense_query

    @property
    def text_query(self) -> str:
        return self.bge_dense_query

    @property
    def keywords(self) -> list[str]:
        return self.english_keywords

class QueryDistiller:
    """
    LLM-powered Query Distillation Module using Google Gemini / OpenRouter API.
    Decomposes raw queries into:
      1. visual_subqueries: Chronological scene visual prompts for SigLIP / TOMS.
      2. bge_dense_query: Dense semantic text for BGE-M3.
      3. english_keywords: Exact sparse keywords for Elasticsearch BM25.
      4. vietnamese_keywords: Vietnamese translation for OCR & subtitle BM25 matching.
    """
    def __init__(self, model: str = None, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_DISTILL_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("DISTILL_MODEL", "google/gemini-2.5-flash")
        self.cache: dict[str, DistilledQueryPayload] = {}

    def extract_terms(self, text: str) -> tuple[list[str], list[str]]:
        """Fallback regex extractor when LLM is offline."""
        cleaned = re.sub(r'[\(\)\[\]\{\}\"\'\,\.\:\;\?\!\-\_\/]', ' ', text)
        words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]
        stop_words = {'the', 'and', 'with', 'from', 'into', 'that', 'this', 'then', 'next', 'over', 'while', 'chef', 'video', 'shows', 'scene'}
        en_kws = [w for w in words if w.lower() not in stop_words][:8]
        return en_kws, []

    def distill(self, query: str, full_context_query: str = None) -> DistilledQueryPayload:
        raw_text = full_context_query or query
        if raw_text in self.cache:
            return self.cache[raw_text]

        if not self.api_key:
            en_kws, vn_kws = self.extract_terms(query)
            payload = DistilledQueryPayload(
                visual_subqueries=[query],
                bge_dense_query=query,
                english_keywords=en_kws,
                vietnamese_keywords=vn_kws
            )
            self.cache[raw_text] = payload
            return payload

        system_prompt = (
            "You are an expert AI video search query optimizer. Given a user search query, decompose it into JSON with 4 keys:\n"
            "1. 'visual_subqueries': list of concise chronological visual scene descriptions (1-3 items) for image-text retrieval (SigLIP).\n"
            "2. 'bge_dense_query': a rich semantic paragraph summarizing core actions and visual details for dense text embedding (BGE-M3).\n"
            "3. 'english_keywords': list of high-value English nouns, actions, and entities for BM25 search.\n"
            "4. 'vietnamese_keywords': list of translated Vietnamese nouns, food/dish names, ingredients, or screen text for OCR/subtitles.\n\n"
            "Respond ONLY with valid JSON."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {raw_text}"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }

        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=15)
            if resp.status_code == 200:
                out = resp.json()
                content = out["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                payload = DistilledQueryPayload(
                    visual_subqueries=parsed.get("visual_subqueries", [query]),
                    bge_dense_query=parsed.get("bge_dense_query", query),
                    english_keywords=parsed.get("english_keywords", []),
                    vietnamese_keywords=parsed.get("vietnamese_keywords", [])
                )
                self.cache[raw_text] = payload
                return payload
        except Exception as e:
            print(f"Distillation API fallback ({e})")

        en_kws, vn_kws = self.extract_terms(query)
        payload = DistilledQueryPayload(
            visual_subqueries=[query],
            bge_dense_query=query,
            english_keywords=en_kws,
            vietnamese_keywords=vn_kws
        )
        self.cache[raw_text] = payload
        return payload
