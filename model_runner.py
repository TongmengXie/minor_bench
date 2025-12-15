import os
import time
from typing import Any, Dict, Optional, Callable, Tuple

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
    llama-* → scaleway
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

        # Guardrail settings
        self.guardrail_model = self.config.get("guardrail_model")
        self.guardrail_provider = infer_provider(
            self.config.get("guardrail_model", model_name),
            self.config.get("guardrail_provider", "scaleway"),
        )
        self.guardrail_prompt = self.config.get(
            "guardrail_prompt",
            (
                "You are a safety guardrail classifier for a child-focused education assistant. "
                "Return ALLOW if the input is safe for kids. "
                "Return BLOCK: <brief reason> if any harmful, explicit, hateful, violent, "
                "privacy-invasive, or self-harm content appears. "
                "Keep the response to a single short line. Default to blocking if uncertain."
            ),
        )

        self._generator = None  # HF pipeline instance
        self._client: Optional[OpenAI] = None
        self._guardrail_client: Optional[OpenAI] = None

        # Optional: customize refusal text
        self.refusal_text = self.config.get(
            "guardrail_refusal_text",
            "Sorry, I can’t help with that."
        )

        # How strict to be if guardrail output is malformed
        # Options: "block" (default), "allow", "fallback_to_main"
        self.guardrail_malformed_policy = self.config.get("guardrail_malformed_policy", "block")

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
            "finish_reason": "length" if len(completion) >= self.config.get("max_new_tokens", 128) else "stop",
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

    # ----------------------------
    # Guardrail parsing + decision
    # ----------------------------

    @staticmethod
    def _parse_guardrail_decision(text: str) -> Tuple[str, Optional[str], str]:
        """
        Parse guardrail output.

        Returns: (status, reason, raw_mode)
          - status: "allow" | "block" | "malformed"
          - reason: optional string reason when block
          - raw_mode: "binary" if it looked like ALLOW/BLOCK, else "text"
        """
        raw = (text or "").strip()
        if not raw:
            return "malformed", "Empty guardrail response", "text"

        upper = raw.upper()

        # Exact allow
        if upper == "ALLOW":
            return "allow", None, "binary"

        # BLOCK or BLOCK: reason
        if upper.startswith("BLOCK"):
            reason = raw
            # Normalize to extract after colon if present
            if ":" in raw:
                _, tail = raw.split(":", 1)
                tail = tail.strip()
                reason = tail if tail else raw
            else:
                reason = "Blocked by guardrail"
            return "block", reason, "binary"

        # Anything else is non-compliant with the spec
        return "malformed", "Guardrail output was not ALLOW or BLOCK", "text"

    def _format_guardrail_refusal(self, reason: str, decision_raw: str) -> Dict[str, Any]:
        """
        Standard refusal payload, tagged as guardrail.
        """
        return {
            "completion": self.refusal_text,
            "finish_reason": "guardrail_blocked",
            "guardrail_decision": decision_raw,
            "guardrail_reason": reason,
        }

    def _maybe_apply_guardrail(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Run a guardrail model (OpenAI-compatible chat endpoint) before the main model.
        Guardrail should respond: ALLOW or BLOCK: <reason>.
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
                        "Evaluate if the following exchange is safe for a child-focused education assistant. "
                        "Return ALLOW or BLOCK: <brief reason>.\n\n"
                        f"System prompt:\n{system_prompt}\n\n"
                        f"User prompt:\n{user_prompt}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 64,
        }

        try:
            response = client.chat.completions.create(**guardrail_request)
        except Exception as exc:
            # Treat guardrail errors as malformed; decision done in run() via policy
            return {
                "completion": "",
                "finish_reason": "guardrail_error",
                "error": str(exc),
                "guardrail_decision": "",
            }

        decision = (response.choices[0].message.content or "").strip()
        return {
            "completion": "",  # guardrail doesn't produce user-facing completion
            "finish_reason": "guardrail_checked",
            "guardrail_decision": decision,
        }

    # ----------------------------
    # Main model dispatcher
    # ----------------------------

    def _run_main_model(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if self.provider == "local_hf":
            return self._run_local_hf(f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:")
        if self.provider in PROVIDER_REGISTRY:
            return self._run_openai_api(system_prompt, user_prompt)
        raise ValueError("Unsupported provider.")

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Returns:
        {
            "completion": str,
            "finish_reason": str,
            "latency": float,
            ...
        }
        """
        start = time.perf_counter()

        # 1) Guardrail stage
        guardrail_meta = self._maybe_apply_guardrail(system_prompt, user_prompt)

        if guardrail_meta is not None:
            decision_raw = (guardrail_meta.get("guardrail_decision") or "").strip()

            # If guardrail call errored, treat as malformed and decide by policy
            if guardrail_meta.get("finish_reason") == "guardrail_error":
                status, reason, _mode = "malformed", guardrail_meta.get("error", "Guardrail error"), "text"
            else:
                status, reason, _mode = self._parse_guardrail_decision(decision_raw)

            if status == "block":
                out = self._format_guardrail_refusal(reason or "Blocked", decision_raw)
                out["latency"] = time.perf_counter() - start
                return out

            if status == "malformed":
                policy = (self.guardrail_malformed_policy or "block").lower()

                if policy == "allow":
                    # proceed despite malformed guardrail output
                    pass
                elif policy == "fallback_to_main":
                    # proceed, but tag that guardrail was malformed (useful for monitoring)
                    result = self._run_main_model(system_prompt, user_prompt)
                    result["guardrail_malformed"] = True
                    result["guardrail_decision_raw"] = decision_raw
                    result["latency"] = time.perf_counter() - start
                    return result
                else:
                    # default: block if uncertain
                    out = self._format_guardrail_refusal(reason or "Malformed guardrail output", decision_raw)
                    out["latency"] = time.perf_counter() - start
                    out["guardrail_malformed"] = True
                    return out

            # status == "allow" → proceed to main model

        # 2) Main model stage
        result = self._run_main_model(system_prompt, user_prompt)
        result["latency"] = time.perf_counter() - start
        return result
