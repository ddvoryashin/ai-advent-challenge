import json
from openai import OpenAI
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "credentials.json"
OUTPUT_FILE = Path(__file__).parent / "task4_results.txt"


def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""


client = OpenAI(
    api_key=load_api_key(),
    base_url="https://api.deepseek.com",
)

QUESTION = "Name one programming language and explain in one sentence why a beginner should learn it."
TEMPERATURES = [0, 0.7, 1.2]
RUNS_PER_TEMP = 3


def query(temperature: float) -> str:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": QUESTION},
        ],
        temperature=temperature,
        max_tokens=100,
        stream=False,
    )
    return resp.choices[0].message.content.strip()


results: dict[float, list[str]] = {}

for temp in TEMPERATURES:
    results[temp] = [query(temp) for _ in range(RUNS_PER_TEMP)]

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    f.write(f"Question: {QUESTION}\n")
    f.write(f"Runs per temperature: {RUNS_PER_TEMP}\n")
    f.write("=" * 60 + "\n\n")

    for temp in TEMPERATURES:
        f.write(f"TEMPERATURE = {temp}\n")
        f.write("-" * 40 + "\n")
        for i, answer in enumerate(results[temp], 1):
            f.write(f"Run {i}: {answer}\n")
        f.write("\n")
