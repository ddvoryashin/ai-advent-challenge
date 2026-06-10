import os
import json
from openai import OpenAI
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "credentials.json"
 
 
def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""

client = OpenAI(
    api_key=load_api_key(),
    base_url="https://api.deepseek.com")

QUESTION = "Explain what Trino is"

# Запрос без ограничений
resp_free = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": QUESTION},
    ],
    stream=False,
    reasoning_effort="low",
    extra_body={"thinking": {"type": "enabled"}}
)

# Запрос с ограничениями
resp_limited = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an experienced data architect. "
                "Be as laconic as you can.\n"
                "Describe pros and cons of the software compared to default choices or closest alternatives.\n"
                "Make the comparison in a numerical format, so that the degree of difference is clear.\n"
                "Your goal is to write an answer that will be understood in 30 sec reading\n"
                "and will give as musch understanding as possible.\n"
                "Answer in Russian.\n"
                "Always add section KEY TAKEAWAY in the end"
            ),
        },
        {"role": "user", "content": QUESTION},
    ],
    max_tokens=500,                                    # жёсткий лимит токенов
    stop=["KEY TAKEAWAY", "Ключевые особенности"],      # остановиться перед финальным блоком
    stream=False,
)


print("=" * 50)
print("БЕЗ ОГРАНИЧЕНИЙ")
print("=" * 50)
print(resp_free.choices[0].message.content)
print(f"\n[токенов использовано: {resp_free.usage.completion_tokens}]")

print("\n" + "=" * 50)
print("С ОГРАНИЧЕНИЯМИ")
print("=" * 50)
print(resp_limited.choices[0].message.content)
print(f"\n[токенов использовано: {resp_limited.usage.completion_tokens}]")
print(f"[причина остановки: {resp_limited.choices[0].finish_reason}]")