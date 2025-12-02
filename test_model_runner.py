from model_runner import ModelRunner
from dotenv import find_dotenv, load_dotenv
import os
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
        config={"api_key": SCW_SECRET_KEY}  # or use env OPENAI_API_KEY / SCW_SECRET_KEY
    )
    print(f"provider: {runner.provider}")
    system_prompt = "You are a helpful assistant."
    user_prompt = "Say 'OK' and nothing else."

    result = runner.run(system_prompt, user_prompt)
    print("RESULT:", result)


if __name__ == "__main__":
    main()
