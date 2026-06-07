import json
from openai import OpenAI
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "credentials.json"
OUTPUT_FILE = Path(__file__).parent / "task3_results.txt"


def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""


client = OpenAI(
    api_key=load_api_key(),
    base_url="https://api.deepseek.com",
)

TASK = (
    "Write an algorithm to determine whether a string is a palindrome, "
    "ignoring spaces, punctuation and case. "
    'Examples: "A man, a plan, a canal: Panama" -> true, "race a car" -> false.'
)

MODEL = "deepseek-chat"


def chat(system: str, user: str, max_tokens: int = 600) -> tuple[str, int]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        stream=False,
    )
    return resp.choices[0].message.content, resp.usage.completion_tokens


# -- 1. Direct answer ---------------------------------------------------------
ans1, tok1 = chat(
    system="You are a helpful assistant.",
    user=TASK,
)

# -- 2. Step-by-step ----------------------------------------------------------
ans2, tok2 = chat(
    system="You are a helpful assistant.",
    user=f"Solve step by step:\n\n{TASK}",
)

# -- 3. Meta-prompt: first craft a prompt, then solve -------------------------
meta_prompt, _ = chat(
    system="You are a prompt engineering expert.",
    user=(
        f"Write the best possible prompt to solve the following task:\n\n{TASK}\n\n"
        "Return only the prompt text, nothing else."
    ),
    max_tokens=300,
)

ans3, tok3 = chat(
    system="You are a helpful assistant.",
    user=meta_prompt,
)

# -- 4. Expert panel ----------------------------------------------------------
EXPERTS_SYSTEM = (
    "You are facilitating a panel of three experts discussing a programming task. "
    "Each expert gives their own answer:\n"
    "- ANALYST: focuses on correctness and edge cases\n"
    "- ENGINEER: focuses on clean, idiomatic code\n"
    "- CRITIC: points out flaws and suggests improvements\n\n"
    "Format your response exactly as:\n"
    "ANALYST:\n<answer>\n\nENGINEER:\n<answer>\n\nCRITIC:\n<answer>"
)

ans4, tok4 = chat(
    system=EXPERTS_SYSTEM,
    user=TASK,
    max_tokens=900,
)

# -- Write results to file ----------------------------------------------------
sections = [
    ("1. DIRECT ANSWER", ans1, tok1),
    ("2. STEP-BY-STEP", ans2, tok2),
    ("3. META-PROMPT -> SOLUTION", ans3, tok3),
    ("4. EXPERT PANEL", ans4, tok4),
]

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    f.write(f"TASK:\n{TASK}\n\n")
    for title, answer, tokens in sections:
        f.write("=" * 60 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        f.write(answer + "\n")
        f.write(f"\n[tokens: {tokens}]\n\n")

print(f"Results saved to {OUTPUT_FILE}")
