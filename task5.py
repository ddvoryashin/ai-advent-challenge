import json
import time
from openai import OpenAI
from pathlib import Path
from datetime import datetime

CONFIG_FILE = Path(__file__).parent / "credentials.json"
OUTPUT_FILE = Path(__file__).parent / "task5_results.txt"


def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""


client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")

QUESTION = (
    "Пять человек: Алёша, Боря, Вася, Гоша, Дима. Ровно двое из них всегда лгут, "
    "остальные говорят правду. "
    "Алёша: 'Боря лжёт'. "
    "Боря: 'Вася и Гоша говорят правду'. "
    "Вася: 'Алёша лжёт'. "
    "Гоша: 'Среди нас не меньше трёх лжецов'. "
    "Дима: 'Алёша и Вася говорят одно и то же'. "
    "Кто лжецы? Покажи полное рассуждение."
)

MODELS = [
    {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek-V4-Flash (light)",
        "input_price_per_m": 0.07,
        "output_price_per_m": 0.28,
    },
    {
        "id": "deepseek-chat",
        "name": "DeepSeek-V3 (standard)",
        "input_price_per_m": 0.27,
        "output_price_per_m": 1.10,
    },
    {
        "id": "deepseek-reasoner",
        "name": "DeepSeek-R1 (reasoning)",
        "input_price_per_m": 0.55,
        "output_price_per_m": 2.19,
    },
]


def query_model(model: dict) -> dict:
    start = time.time()
    resp = client.chat.completions.create(
        model=model["id"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": QUESTION},
        ],
        stream=False,
    )
    elapsed = time.time() - start
    input_tok = resp.usage.prompt_tokens
    output_tok = resp.usage.completion_tokens
    cost = (
        input_tok / 1_000_000 * model["input_price_per_m"]
        + output_tok / 1_000_000 * model["output_price_per_m"]
    )
    return {
        "answer": resp.choices[0].message.content,
        "elapsed": elapsed,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "cost_usd": cost,
    }


def main():
    lines = [
        f"DeepSeek Model Comparison — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        f"ВОПРОС: {QUESTION}",
        "=" * 70,
    ]
    for model in MODELS:
        lines += [
            f"\n{'=' * 70}",
            f"МОДЕЛЬ: {model['name']} ({model['id']})",
            "=" * 70,
        ]
        r = query_model(model)
        lines += [
            r["answer"],
            "\n--- Метрики ---",
            f"Время ожидания:  {r['elapsed']:.2f} с",
            f"Токены входные:  {r['input_tokens']}",
            f"Токены выходные: {r['output_tokens']}",
            f"Стоимость:       ${r['cost_usd']:.6f}",
        ]
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
