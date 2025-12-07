import os
import time
from typing import Any, Dict, Optional, Callable

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

ProviderFactory = Callable[[Dict[str, Any]], OpenAI]


# ------------------------------------------------------------
# Provider builders
# ------------------------------------------------------------

def _build_openai_client(config: Dict[str, Any]) -> OpenAI:
    api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for OpenAI provider")

    base_url = config.get("api_base") or os.getenv("OPENAI_BASE_URL") or None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def _build_scaleway_client(config: Dict[str, Any]) -> OpenAI:
    api_key = config.get("api_key") or os.getenv("SCW_SECRET_KEY")
    if not api_key:
        raise RuntimeError("Missing SCW_SECRET_KEY for Scaleway provider")

    base_url = config.get("api_base") or os.getenv("SCW_BASE_URL") or "https://api.scaleway.ai/v1"
    return OpenAI(api_key=api_key, base_url=base_url)


PROVIDER_REGISTRY: Dict[str, ProviderFactory] = {
    "openai": _build_openai_client,
    "scaleway": _build_scaleway_client,
}


# ------------------------------------------------------------
# Provider inference
# ------------------------------------------------------------

def infer_provider(model_name: str, provider: Optional[str] = None) -> str:
    """
    Normalized provider selection logic.
    If provider="auto", choose based on model.
    llama-3.1 → scaleway
    everything else → openai
    """

    requested = (provider or "auto").lower()

    if requested in {"local_hf"}:
        return requested
    if requested == "openai_api":
        return "openai"
    if requested in PROVIDER_REGISTRY:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported provider '{provider}'.")

    # infer automatically
    if "llama" in model_name.lower():
        return "scaleway"

    return "openai"


# ------------------------------------------------------------
# Main ModelRunner class
# ------------------------------------------------------------

class ModelRunner:
    """
    Unified interface for running local HF models or API models.
    """

    def __init__(self, model_name: str, provider: Optional[str], config: Optional[Dict[str, Any]] = None):
        load_dotenv(find_dotenv())

        self.model_name = model_name
        self.provider = infer_provider(model_name, provider)
        self.config = config or {}
        self.guardrail_model = self.config.get("guardrail_model")
        self.guardrail_provider = infer_provider(
            self.config.get("guardrail_model", model_name),
            self.config.get("guardrail_provider", "scaleway"),
        )
        self.guardrail_prompt = self.config.get(
            "guardrail_prompt",
            (
                "You are a safety guardrail classifier. Return 'ALLOW' if the input is safe for a child-focused "
                "education bot. Otherwise return 'BLOCK: <reason>'. Keep responses concise."
            ),
        )
        self._generator = None  # HF pipeline instance
        self._client: Optional[OpenAI] = None
        self._guardrail_client: Optional[OpenAI] = None
    # ----------------------------
    # Local HF
    # ----------------------------

    def _init_local_hf(self) -> None:
        if self._generator is not None:
            return

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Transformers must be installed for local_hf provider.") from exc

        generation_kwargs = {"model": self.model_name}
        generation_kwargs.update(self.config.get("generation_kwargs", {}))

        self._generator = pipeline("text-generation", **generation_kwargs)

    def _run_local_hf(self, prompt: str) -> Dict[str, Any]:
        self._init_local_hf()
        outputs = self._generator(prompt, max_new_tokens=self.config.get("max_new_tokens", 128))
        completion = outputs[0].get("generated_text", "").replace(prompt, "", 1).strip()

        return {
            "completion": completion,
            "finish_reason": (
                "length" if len(completion) >= self.config.get("max_new_tokens", 128) else "stop"
            ),
        }

    # ----------------------------
    # API clients
    # ----------------------------

    def _get_client(self) -> OpenAI:
        if self.provider == "local_hf":
            raise RuntimeError("Local HF provider does not use an API client")

        if self._client:
            return self._client

        # Remove proxies that break requests
        for key in [
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
            "OPENAI_PROXY",
        ]:
            if key in os.environ:
                del os.environ[key]

        factory = PROVIDER_REGISTRY.get(self.provider)
        if not factory:
            raise ValueError(f"Unsupported provider '{self.provider}'.")

        self._client = factory(self.config)
        return self._client
    
    def _get_guardrail_client(self) -> Optional[OpenAI]:
        if not self.guardrail_model:
            return None

        if self._guardrail_client:
            return self._guardrail_client

        factory = PROVIDER_REGISTRY.get(self.guardrail_provider)
        if not factory:
            raise ValueError(f"Unsupported guardrail provider '{self.guardrail_provider}'.")

        guardrail_config = dict(self.config)
        guardrail_config.setdefault("model_name", self.guardrail_model)

        self._guardrail_client = factory(guardrail_config)
        return self._guardrail_client
    
    def _run_openai_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        client = self._get_client()
        request_kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.get("temperature", 0.2),
            "max_tokens": self.config.get("max_tokens", 256),
        }
        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            return {"completion": "", "finish_reason": "error", "error": str(exc)}

        content = response.choices[0].message.content or ""
        return {
            "completion": content,
            "finish_reason": response.choices[0].finish_reason,
        }
    
    def _maybe_apply_guardrail(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Run a guardrail model (via the OpenAI-compatible chat endpoint) before the
        main model call. Guardrail models run on Scaleway by default and should
        respond with either "ALLOW" or "BLOCK: <reason>".
        """

        if not self.guardrail_model:
            return None

        client = self._get_guardrail_client()
        if not client:
            return None

        guardrail_request = {
            "model": self.guardrail_model,
            "messages": [
                {"role": "system", "content": self.guardrail_prompt},
                {
                    "role": "user",
                    "content": (
                        "Evaluate if the following exchange is safe for an education bot aimed at children. "
                        "Return ALLOW or BLOCK with a brief reason.\n\n"
                        f"System prompt: {system_prompt}\nUser prompt: {user_prompt}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 64,
        }

        try:
            response = client.chat.completions.create(**guardrail_request)
        except Exception as exc:
            return {"completion": "", "finish_reason": "guardrail_error", "error": str(exc)}

        decision = (response.choices[0].message.content or "").strip()
        if decision.upper().startswith("BLOCK"):
            return {
                "completion": "",
                "finish_reason": "guardrail_blocked",
                "guardrail_decision": decision,
            }

        return None
    # ----------------------------
    # Public API
    # ----------------------------

    def run(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Returns:
        {
            "completion": str,
            "finish_reason": str,
            "latency": float
        }
        """
        start = time.perf_counter()
        
        guardrail_result = self._maybe_apply_guardrail(system_prompt, user_prompt)
        if guardrail_result is not None:
            guardrail_result["latency"] = time.perf_counter() - start
            return guardrail_result
        
        if self.provider == "local_hf":
            result = self._run_local_hf(f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:")
        elif self.provider in PROVIDER_REGISTRY:
            result = self._run_openai_api(system_prompt, user_prompt)
        else:
            raise ValueError("Unsupported provider.")

        result["latency"] = time.perf_counter() - start
        return result
