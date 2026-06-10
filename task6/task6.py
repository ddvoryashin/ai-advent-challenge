import json
from openai import OpenAI
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "credentials.json"
MODEL = "deepseek-chat"
SYSTEM_PROMPT = "You are a helpful assistant. Reply concisely."
MAX_HISTORY_TOKENS = 2000


def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""


def _estimate_tokens(messages: list[dict]) -> int:
    # ~4 chars per token + 4 tokens of framing overhead per message
    return sum(len(m["content"]) // 4 + 4 for m in messages)


class Agent:
    """Conversational agent with a token-budget sliding window over history."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        max_history_tokens: int = MAX_HISTORY_TOKENS,
    ) -> None:
        self._client = client
        self._model = model
        self._max_history_tokens = max_history_tokens
        self._history: list[dict] = [{"role": "system", "content": system_prompt}]

    def _trim(self) -> int:
        """Drop oldest non-system messages until history fits the token budget.
        Always keeps system prompt + the latest user message."""
        dropped = 0
        while (
            _estimate_tokens(self._history) > self._max_history_tokens
            and len(self._history) > 2
        ):
            self._history.pop(1)
            dropped += 1
        return dropped

    def chat(self, user_message: str) -> tuple[str, int]:
        """Returns (reply, number_of_messages_trimmed)."""
        self._history.append({"role": "user", "content": user_message})
        dropped = self._trim()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._history,
        )
        reply = response.choices[0].message.content
        self._history.append({"role": "assistant", "content": reply})
        return reply, dropped

    def reset(self) -> None:
        system = self._history[0]
        self._history = [system]

    def token_usage(self) -> int:
        return _estimate_tokens(self._history)

    def message_count(self) -> int:
        return len(self._history) - 1  # exclude system prompt


def main() -> None:
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")
    agent = Agent(client, MODEL, SYSTEM_PROMPT)

    print(f"Agent ready (model: {MODEL}, history limit: ~{MAX_HISTORY_TOKENS} tokens).")
    print("Commands: /reset — clear history, /quit — exit\n")

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

        reply, dropped = agent.chat(user_input)
        print(f"Agent: {reply}")
        status = f"[~{agent.token_usage()} tok | {agent.message_count()} msg"
        if dropped:
            status += f" | trimmed {dropped} old msg(s)"
        print(f"{status}]\n")


if __name__ == "__main__":
    main()
