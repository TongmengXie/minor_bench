from model_runner import ModelRunner

def main():
    runner = ModelRunner(
        model_name="gpt-4o-mini",
        provider="openai_api",
        config={"api_key": ""}  # or leave empty if using env
    )

    system_prompt = "You are a helpful assistant."
    user_prompt = "Say 'OK' and nothing else."

    result = runner.run(system_prompt, user_prompt)
    print("RESULT:", result)

if __name__ == "__main__":
    main()
