import os
import sys
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

# Ensure local imports work when running this file directly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from model_runner import ModelRunner
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SCW_SECRET_KEY = os.getenv('SCW_SECRET_KEY')

def main():
    # runner = ModelRunner(
    #     model_name="gpt-4o-mini",
    #     provider="openai",  # or "auto"
    #     config={"api_key": OPENAI_API_KEY}  # or use env OPENAI_API_KEY / SCW_SECRET_KEY
    # )
    runner = ModelRunner(
        # model_name="gpt-4o-mini",
        model_name="llama-3.1-8b-instruct",
        provider="auto",  # or "auto"
        # config={"api_key": SCW_SECRET_KEY, "guardrail_model":"ibm-granite/granite-guardian-3.2-3b-a800m", "guardrail_provider": "local_hf"}  # or use env OPENAI_API_KEY / SCW_SECRET_KEY
        config={"api_key": SCW_SECRET_KEY, "guardrail_model":"nvidia/llama-3.1-nemoguard-8b-content-safety", "guardrail_provider": "local_hf"}
    )
    print(f"provider: {runner.provider}")
    system_prompt = "You are a helpful assistant."
    user_prompt = "What day is it today?"

    result = runner.run(system_prompt, user_prompt)
    print("RESULT:", result)


if __name__ == "__main__":
    main()
