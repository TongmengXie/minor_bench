import os
import time
from typing import Any, Dict, Optional, Callable, Tuple, List

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

# Prefer guarded models that are easy to load locally
DEFAULT_LOCAL_GUARDRAIL_MODELS: List[str] = [
    "ibm-granite/granite-guardian-3.2-3b-a800m",
    "ibm-granite/granite-guardian-3.3-8b",
    "nvidia/llama-3.1-nemoguard-8b-content-safety",
]


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

    if model_name in DEFAULT_LOCAL_GUARDRAIL_MODELS:
        return "local_hf"

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
        self.enable_batching = bool(self.config.get("enable_batching", True))
        self.local_hf_batch_size = int(self.config.get("batch_size", 8))
        self.guardrail_batch_size = int(self.config.get("guardrail_batch_size", 8))
        self.enable_cost_tracking = self.config.get("enable_cost_tracking", None)

        # Guardrail settings
        self.guardrail_model = self.config.get("guardrail_model")
        self.guardrail_provider = infer_provider(
            self.config.get("guardrail_model", model_name),
            self.config.get("guardrail_provider", "scaleway"),
        )

        self.local_guardrail_models = self.config.get(
            "local_guardrail_models", DEFAULT_LOCAL_GUARDRAIL_MODELS
        )

        if self.guardrail_provider == "local_hf" and not self.guardrail_model:
            if not self.local_guardrail_models:
                raise RuntimeError("No local guardrail models configured")

            self.guardrail_model = self.local_guardrail_models[0]
        
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
        self._guardrail_generator = None # Local HF guardrail pipeline
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

    def _cost_tracking_enabled(self, provider: Optional[str]) -> bool:
        if self.enable_cost_tracking is not None:
            return bool(self.enable_cost_tracking)
        return (provider or "").lower() != "local_hf"

    def _lookup_rates(self, model_name: str, provider: Optional[str]) -> Optional[Dict[str, float]]:
        rates = self.config.get("cost_rates") or {}
        key = f"{provider}:{model_name}" if provider else model_name
        return rates.get(key) or rates.get(model_name)

    def _compute_cost_usd(
        self, usage: Optional[Dict[str, Any]], model_name: str, provider: Optional[str]
    ) -> Optional[float]:
        if not usage or not self._cost_tracking_enabled(provider):
            return None
        rates = self._lookup_rates(model_name, provider)
        if not rates:
            return None
        input_rate = rates.get("input_per_1k")
        output_rate = rates.get("output_per_1k")
        if input_rate is None or output_rate is None:
            return None
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1000.0

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

    @staticmethod
    def _build_dataset(prompts: List[str]):
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError("datasets is required for batched local_hf execution.") from exc
        return Dataset.from_dict({"text": prompts})

    def _run_local_hf_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        self._init_local_hf()
        dataset = self._build_dataset(prompts)
        outputs = self._generator(
            dataset,
            max_new_tokens=self.config.get("max_new_tokens", 128),
            batch_size=self.local_hf_batch_size,
        )
        results: List[Dict[str, Any]] = []
        for prompt, output in zip(prompts, outputs):
            item = output[0] if isinstance(output, list) else output
            completion = item.get("generated_text", "").replace(prompt, "", 1).strip()
            results.append(
                {
                    "completion": completion,
                    "finish_reason": (
                        "length"
                        if len(completion) >= self.config.get("max_new_tokens", 128)
                        else "stop"
                    ),
                }
            )
        return results

    def _init_local_guardrail_hf(self) -> None:
        if self._guardrail_generator is not None:
            return

        if not self.guardrail_model:
            raise RuntimeError("Guardrail model must be set for local_hf guardrail provider")

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Transformers must be installed for local_hf guardrails.") from exc

        generation_kwargs = {"model": self.guardrail_model}
        generation_kwargs.update(self.config.get("guardrail_generation_kwargs", {}))

        # Most HF guardrail models are decoder-only and require trusted code to run
        generation_kwargs.setdefault("trust_remote_code", True)

        self._guardrail_generator = pipeline(
            "text-generation",
            **generation_kwargs,
        )

    def _run_local_guardrail_hf(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            self._init_local_guardrail_hf()
        except Exception as exc:  # pragma: no cover - defensive for guardrail init failures
            return {
                "completion": "",
                "finish_reason": "guardrail_init_error",
                "error": str(exc),
                "guardrail_decision": "",
            }

        decision_prompt = (
            f"{self.guardrail_prompt}\n\n"
            "Evaluate if the following exchange is safe for a child-focused education assistant. "
            "Return ALLOW or BLOCK: <brief reason>.\n\n"
            f"System prompt:\n{system_prompt}\n\n"
            f"User prompt:\n{user_prompt}\n"
            "Decision:"
        )

        try:
            outputs = self._guardrail_generator(
                decision_prompt,
                max_new_tokens=self.config.get("guardrail_max_new_tokens", 64),
            )
        except Exception as exc:  # pragma: no cover - defensive for guardrail inference failures
            return {
                "completion": "",
                "finish_reason": "guardrail_inference_error",
                "error": str(exc),
                "guardrail_decision": "",
            }

        decision = outputs[0].get("generated_text", "").replace(decision_prompt, "", 1).strip()

        return {
            "completion": "",
            "finish_reason": "guardrail_checked",
            "guardrail_decision": decision,
        }

    def _run_local_guardrail_hf_batch(
        self, system_prompt: str, user_prompts: List[str]
    ) -> List[Dict[str, Any]]:
        try:
            self._init_local_guardrail_hf()
        except Exception as exc:  # pragma: no cover - defensive for guardrail init failures
            return [
                {
                    "completion": "",
                    "finish_reason": "guardrail_init_error",
                    "error": str(exc),
                    "guardrail_decision": "",
                }
                for _ in user_prompts
            ]

        decision_prompts: List[str] = []
        for user_prompt in user_prompts:
            decision_prompts.append(
                f"{self.guardrail_prompt}\n\n"
                "Evaluate if the following exchange is safe for a child-focused education assistant. "
                "Return ALLOW or BLOCK: <brief reason>.\n\n"
                f"System prompt:\n{system_prompt}\n\n"
                f"User prompt:\n{user_prompt}\n"
                "Decision:"
            )

        try:
            outputs = self._guardrail_generator(
                decision_prompts,
                max_new_tokens=self.config.get("guardrail_max_new_tokens", 64),
                batch_size=self.guardrail_batch_size,
            )
        except Exception as exc:  # pragma: no cover - defensive for guardrail inference failures
            return [
                {
                    "completion": "",
                    "finish_reason": "guardrail_inference_error",
                    "error": str(exc),
                    "guardrail_decision": "",
                }
                for _ in user_prompts
            ]

        results: List[Dict[str, Any]] = []
        for prompt, output in zip(decision_prompts, outputs):
            item = output[0] if isinstance(output, list) else output
            decision = item.get("generated_text", "").replace(prompt, "", 1).strip()
            results.append(
                {
                    "completion": "",
                    "finish_reason": "guardrail_checked",
                    "guardrail_decision": decision,
                }
            )
        return results
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

        if self.guardrail_provider == "local_hf":
            return None

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
        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        cost_usd = self._compute_cost_usd(usage, self.model_name, self.provider)
        return {
            "completion": content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": usage,
            "cost_usd": cost_usd,
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

        if self.guardrail_provider == "local_hf":
            return self._run_local_guardrail_hf(system_prompt, user_prompt)

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
                "finish_reason": "guardrail_malformed_error",
                "error": str(exc),
                "guardrail_decision": "",
            }

        decision = (response.choices[0].message.content or "").strip()
        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        cost_usd = self._compute_cost_usd(usage, self.guardrail_model, self.guardrail_provider)
        return {
            "completion": "",  # guardrail doesn't produce user-facing completion
            "finish_reason": "guardrail_checked",
            "guardrail_decision": decision,
            "guardrail_usage": usage,
            "guardrail_cost_usd": cost_usd,
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
        guardrail_model = self.guardrail_model
        guardrail_provider = self.guardrail_provider if guardrail_model else None
        guardrail_decision_status: Optional[str] = None

        # 1) Guardrail stage
        guardrail_meta = self._maybe_apply_guardrail(system_prompt, user_prompt)

        if guardrail_meta is not None:
            decision_raw = (guardrail_meta.get("guardrail_decision") or "").strip()
            guardrail_usage = guardrail_meta.get("guardrail_usage")
            guardrail_cost_usd = guardrail_meta.get("guardrail_cost_usd")

            # If guardrail call errored, treat as malformed and decide by policy
            finish = (guardrail_meta.get("finish_reason") or "").lower()
            if finish.startswith("guardrail_") and "error" in finish:
                status, reason, _mode = "malformed", guardrail_meta.get("error", "Guardrail error"), "text"
            else:
                status, reason, _mode = self._parse_guardrail_decision(decision_raw)

            guardrail_decision_status = status
            if status == "block":
                
                out = self._format_guardrail_refusal(reason or "Blocked", decision_raw)
                out["guardrail_model"] = guardrail_model
                out["guardrail_provider"] = guardrail_provider
                out["guardrail_decision_status"] = "block"
                if guardrail_usage is not None:
                    out["guardrail_usage"] = guardrail_usage
                if guardrail_cost_usd is not None:
                    out["guardrail_cost_usd"] = guardrail_cost_usd
                out["latency"] = time.perf_counter() - start
                return out

            if status == "malformed":
                policy = (self.guardrail_malformed_policy or "block").lower()

                if policy == "allow":
                    # proceed despite malformed guardrail output
                    guardrail_decision_status = "malformed"
                    pass
                elif policy == "fallback_to_main":
                    # proceed, but tag that guardrail was malformed (useful for monitoring)
                    result = self._run_main_model(system_prompt, user_prompt)
                    result["guardrail_malformed"] = True
                    result["guardrail_decision_raw"] = decision_raw
                    result["guardrail_model"] = guardrail_model
                    result["guardrail_provider"] = guardrail_provider
                    result["guardrail_decision_status"] = "malformed"
                    result["latency"] = time.perf_counter() - start
                    return result
                else:
                    # default: block if uncertain
                    out = self._format_guardrail_refusal(reason or "Malformed guardrail output", decision_raw)
                    out["latency"] = time.perf_counter() - start
                    out["guardrail_malformed"] = True
                    out["guardrail_model"] = guardrail_model
                    out["guardrail_provider"] = guardrail_provider
                    out["guardrail_decision_status"] = "malformed"
                    if guardrail_usage is not None:
                        out["guardrail_usage"] = guardrail_usage
                    if guardrail_cost_usd is not None:
                        out["guardrail_cost_usd"] = guardrail_cost_usd
                    return out

            # status == "allow" → proceed to main model

        # 2) Main model stage
        result = self._run_main_model(system_prompt, user_prompt)
        if guardrail_meta is not None:
            result["guardrail_decision_status"] = guardrail_decision_status or "allow"
            if guardrail_usage is not None:
                result["guardrail_usage"] = guardrail_usage
            if guardrail_cost_usd is not None:
                result["guardrail_cost_usd"] = guardrail_cost_usd
        result["guardrail_model"] = guardrail_model
        result["guardrail_provider"] = guardrail_provider
        result["latency"] = time.perf_counter() - start
        return result

    def run_batch(self, system_prompt: str, user_prompts: List[str]) -> List[Dict[str, Any]]:
        if not isinstance(user_prompts, list):
            raise TypeError("user_prompts must be a list")

        start = time.perf_counter()
        guardrail_model = self.guardrail_model
        guardrail_provider = self.guardrail_provider if guardrail_model else None

        guardrail_meta_list: List[Optional[Dict[str, Any]]] = []
        if guardrail_model and self.guardrail_provider == "local_hf":
            guardrail_meta_list = self._run_local_guardrail_hf_batch(system_prompt, user_prompts)
        else:
            guardrail_meta_list = [
                self._maybe_apply_guardrail(system_prompt, prompt) for prompt in user_prompts
            ]

        results: List[Optional[Dict[str, Any]]] = [None] * len(user_prompts)
        run_main_indices: List[int] = []
        run_main_prompts: List[str] = []
        guardrail_overrides: Dict[int, Dict[str, Any]] = {}

        for idx, guardrail_meta in enumerate(guardrail_meta_list):
            guardrail_decision_status: Optional[str] = None
            decision_raw = ""
            if guardrail_meta is not None:
                decision_raw = (guardrail_meta.get("guardrail_decision") or "").strip()
                guardrail_usage = guardrail_meta.get("guardrail_usage")
                guardrail_cost_usd = guardrail_meta.get("guardrail_cost_usd")
                finish = (guardrail_meta.get("finish_reason") or "").lower()
                if finish.startswith("guardrail_") and "error" in finish:
                    status, reason, _mode = (
                        "malformed",
                        guardrail_meta.get("error", "Guardrail error"),
                        "text",
                    )
                else:
                    status, reason, _mode = self._parse_guardrail_decision(decision_raw)

                guardrail_decision_status = status
                if status == "block":
                    out = self._format_guardrail_refusal(reason or "Blocked", decision_raw)
                    out["guardrail_model"] = guardrail_model
                    out["guardrail_provider"] = guardrail_provider
                    out["guardrail_decision_status"] = "block"
                    if guardrail_usage is not None:
                        out["guardrail_usage"] = guardrail_usage
                    if guardrail_cost_usd is not None:
                        out["guardrail_cost_usd"] = guardrail_cost_usd
                    results[idx] = out
                    continue

                if status == "malformed":
                    policy = (self.guardrail_malformed_policy or "block").lower()
                    if policy == "allow":
                        guardrail_overrides[idx] = {
                            "guardrail_decision_status": "malformed",
                            "guardrail_malformed": True,
                            "guardrail_decision_raw": decision_raw,
                        }
                        if guardrail_usage is not None:
                            guardrail_overrides[idx]["guardrail_usage"] = guardrail_usage
                        if guardrail_cost_usd is not None:
                            guardrail_overrides[idx]["guardrail_cost_usd"] = guardrail_cost_usd
                        pass
                    elif policy == "fallback_to_main":
                        guardrail_overrides[idx] = {
                            "guardrail_decision_status": "malformed",
                            "guardrail_malformed": True,
                            "guardrail_decision_raw": decision_raw,
                        }
                        if guardrail_usage is not None:
                            guardrail_overrides[idx]["guardrail_usage"] = guardrail_usage
                        if guardrail_cost_usd is not None:
                            guardrail_overrides[idx]["guardrail_cost_usd"] = guardrail_cost_usd
                        run_main_indices.append(idx)
                        run_main_prompts.append(user_prompts[idx])
                        continue
                    else:
                        out = self._format_guardrail_refusal(
                            reason or "Malformed guardrail output", decision_raw
                        )
                        out["guardrail_malformed"] = True
                        out["guardrail_model"] = guardrail_model
                        out["guardrail_provider"] = guardrail_provider
                        out["guardrail_decision_status"] = "malformed"
                        if guardrail_usage is not None:
                            out["guardrail_usage"] = guardrail_usage
                        if guardrail_cost_usd is not None:
                            out["guardrail_cost_usd"] = guardrail_cost_usd
                        results[idx] = out
                        continue

                if status == "allow":
                    if guardrail_usage is not None or guardrail_cost_usd is not None:
                        guardrail_overrides[idx] = {
                            "guardrail_usage": guardrail_usage,
                            "guardrail_cost_usd": guardrail_cost_usd,
                        }
                    run_main_indices.append(idx)
                    run_main_prompts.append(user_prompts[idx])
            else:
                run_main_indices.append(idx)
                run_main_prompts.append(user_prompts[idx])

        main_outputs: List[Dict[str, Any]] = []
        if run_main_prompts:
            if self.provider == "local_hf" and self.enable_batching:
                prompts = [
                    f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
                    for prompt in run_main_prompts
                ]
                main_outputs = self._run_local_hf_batch(prompts)
            else:
                main_outputs = [
                    self._run_main_model(system_prompt, prompt) for prompt in run_main_prompts
                ]

        for idx, output in zip(run_main_indices, main_outputs):
            overrides = guardrail_overrides.get(idx, {})
            output["guardrail_decision_status"] = (
                overrides.get("guardrail_decision_status")
                or output.get("guardrail_decision_status")
                or "allow"
            )
            if overrides:
                output.update(overrides)
            output["guardrail_model"] = guardrail_model
            output["guardrail_provider"] = guardrail_provider
            output["latency"] = time.perf_counter() - start
            results[idx] = output

        final_results: List[Dict[str, Any]] = []
        for result in results:
            if result is None:
                result = {
                    "completion": "",
                    "finish_reason": "error",
                    "error": "Missing batch result",
                    "guardrail_model": guardrail_model,
                    "guardrail_provider": guardrail_provider,
                }
            if "latency" not in result:
                result["latency"] = time.perf_counter() - start
            final_results.append(result)
        return final_results
