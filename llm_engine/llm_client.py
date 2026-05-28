# llm_core/llm_client.py
from __future__ import annotations

import json
import requests
from typing import Any, Dict, List, Optional


class OpenRouterClient:
    """
    Minimal OpenRouter Chat Completions client.
    Stores raw_request/raw_response so you can save them in DB.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 60,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.site_url = site_url
        self.app_name = app_name

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra:
            payload.update(extra)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Optional but recommended by OpenRouter
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
