import os
import time, httpx
from typing import Any, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


class ModelRunner:
    """
    Unified interface for running local Hugging Face models or OpenAI-like APIs.
    """

    def __init__(self, model_name: str, provider: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.provider = provider
        self.config = config or {}
        self._generator = None
        load_dotenv(find_dotenv())

    def _init_local_hf(self) -> None:
        if self._generator is not None:
            return

        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError(
                "Transformers must be installed to use provider='local_hf'."
            ) from exc

        generation_kwargs = {
            "model": self.model_name,
            **self.config.get("generation_kwargs", {}),
        }
        self._generator = pipeline("text-generation", **generation_kwargs)

    def _run_local_hf(self, prompt: str) -> Dict[str, Any]:
        self._init_local_hf()
        assert self._generator is not None
        outputs = self._generator(prompt, max_new_tokens=self.config.get("max_new_tokens", 128))
        completion = outputs[0].get("generated_text", "").replace(prompt, "", 1).strip()
        return {
            "completion": completion,
            "finish_reason": "length" if len(completion) >= self.config.get("max_new_tokens", 128) else "stop",
        }

    def _run_openai_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")

        base_url = (
            self.config.get("api_base") or
            os.getenv("OPENAI_BASE_URL") or
            None
        )

        # organization = (
        #     self.config.get("organization") or
        #     os.getenv("OPENAI_ORG") or
        #     os.getenv("OPENAI_ORGANIZATION") or
        #     None
        # )

        # Build client exactly as official docs suggest
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        # if organization:
        #     client_kwargs["organization"] = organization

        # REMOVE ALL PROXY ENV VARS
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                    "http_proxy", "https_proxy", "all_proxy",
                    "OPENAI_PROXY"]:
            if key in os.environ:
                del os.environ[key]

        client = OpenAI(**client_kwargs)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.get("temperature", 0.2),
                max_tokens=self.config.get("max_tokens", 256),
            )
        except Exception as exc:
            return {
                "completion": "",
                "finish_reason": "error",
                "error": str(exc),
            }

        content = response.choices[0].message.content or ""

        return {
            "completion": content,
            "finish_reason": response.choices[0].finish_reason,
        }

    def run(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Returns dict:
        {
          "completion": <string>,
          "latency": <float>,
          "finish_reason": <string>
        }
        """

        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        start = time.perf_counter()
        if self.provider == "local_hf":
            result = self._run_local_hf(full_prompt)
        elif self.provider == "openai_api":
            result = self._run_openai_api(system_prompt, user_prompt)
        else:
            raise ValueError("Unsupported provider. Use 'local_hf' or 'openai_api'.")
        latency = time.perf_counter() - start
        result["latency"] = latency
        return result