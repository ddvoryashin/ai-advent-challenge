import json
from openai import OpenAI
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "credentials.json"
HISTORY_FILE = Path(__file__).parent / "history.jsonl"
MODEL = "deepseek-chat"
SYSTEM_PROMPT = (
    "You are an experienced architect and data engineer with 15 years of experience. "
    "Reply concisely, be professional, keep it short and highlight what is important for data and business in general. "
    "Think about technical and business context of your answers and solutions"
)
WINDOW_TOKENS = 2000


def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m["content"]) // 4 + 4 for m in messages)


def _read_history(history_file: Path) -> list[dict]:
    messages = []
    if history_file.exists():
        with history_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return messages


def _append_messages(history_file: Path, *messages: dict) -> None:
    with history_file.open("a", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def history_stats(history_file: Path, system: dict) -> tuple[int, int]:
    """Returns (message_count, estimated_total_tokens including system prompt)."""
    messages = _read_history(history_file)
    tokens = _estimate_tokens([system] + messages)
    return len(messages), tokens


class Agent:
    """Conversational agent with append-only persistent history and a sliding-window prompt."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        history_file: Path,
        window_tokens: int = WINDOW_TOKENS,
    ) -> None:
        self._client = client
        self._model = model
        self._system = {"role": "system", "content": system_prompt}
        self._history_file = history_file
        self._window_tokens = window_tokens

    def _build_window(self) -> list[dict]:
        """System prompt + as many recent messages as fit within window_tokens."""
        budget = self._window_tokens - _estimate_tokens([self._system])
        window: list[dict] = []
        for msg in reversed(_read_history(self._history_file)):
            cost = len(msg["content"]) // 4 + 4
            if cost > budget:
                break
            budget -= cost
            window.insert(0, msg)
        return [self._system] + window

    def chat(self, user_message: str) -> tuple[str, int]:
        """Returns (reply, messages_excluded_from_window)."""
        user_msg = {"role": "user", "content": user_message}
        _append_messages(self._history_file, user_msg)
        messages = self._build_window()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        reply = response.choices[0].message.content
        assistant_msg = {"role": "assistant", "content": reply}
        _append_messages(self._history_file, assistant_msg)
        total = history_stats(self._history_file, self._system)[0]
        excluded = max(0, total - (len(messages) - 1))
        return reply, excluded

    def reset(self) -> None:
        self._history_file.unlink(missing_ok=True)

    def window_tokens(self) -> int:
        return _estimate_tokens(self._build_window())


def main() -> None:
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")
    agent = Agent(client, MODEL, SYSTEM_PROMPT, HISTORY_FILE)

    existing, _ = history_stats(HISTORY_FILE, {"role": "system", "content": SYSTEM_PROMPT})
    if existing:
        print(f"Resuming session — {existing} message(s) in {HISTORY_FILE.name}")
    else:
        print("New session started.")
    print(f"Model: {MODEL} | window limit: ~{WINDOW_TOKENS} tokens")
    print("Commands: /history — stats, /reset — clear history, /quit — exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            print("Bye!")
            break
        if user_input == "/reset":
            agent.reset()
            print("--- history cleared ---\n")
            continue
        if user_input == "/history":
            count, tokens = history_stats(HISTORY_FILE, {"role": "system", "content": SYSTEM_PROMPT})
            print(f"[history: {count} messages | ~{tokens} tokens total]\n")
            continue

        reply, excluded = agent.chat(user_input)
        print(f"Agent: {reply}")
        status = f"[~{agent.window_tokens()} tok in window"
        if excluded:
            status += f" | {excluded} old msg(s) outside window"
        print(f"{status}]\n")


if __name__ == "__main__":
    main()
