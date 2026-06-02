import os
import json
from openai import OpenAI
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "credentials.json"
 
 
def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""

client = OpenAI(
    api_key=load_api_key(),
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False,
    reasoning_effort="low",
    extra_body={"thinking": {"type": "enabled"}}
)

print(response.choices[0].message.content)