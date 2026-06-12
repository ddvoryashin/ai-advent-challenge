"""
Context management agent: keeps last RECENT_KEEP messages verbatim,
summarizes older messages in chunks and stores the summary separately.

Context sent to the model = system + [summary injection] + recent N messages.
"""

import json
from dataclasses import dataclass
from openai import OpenAI
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "credentials.json"
HISTORY_FILE = Path(__file__).parent / "history.jsonl"
SUMMARY_FILE = Path(__file__).parent / "summary.json"
MODEL = "deepseek-chat"
RECENT_KEEP = 5     # messages kept verbatim in every request
SUMMARY_CHUNK = 10  # trigger summarization when this many messages accumulate outside the window
SYSTEM_PROMPT = (
    "You are an experienced architect and data engineer with 15 years of experience. "
    "Reply concisely, be professional, keep it short and highlight what is important for data and business in general. "
    "Think about technical and business context of your answers and solutions"
)


def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m["content"]) // 4 + 4 for m in messages)


def _read_history(path: Path) -> list[dict]:
    msgs: list[dict] = []
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        msgs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return msgs


def _append_messages(path: Path, *messages: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def _load_summary(path: Path) -> dict:
    """Returns {"content": str, "covered_count": int} or empty defaults."""
    if path.exists():
        with path.open() as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {"content": "", "covered_count": 0}


def _save_summary(path: Path, content: str, covered_count: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({"content": content, "covered_count": covered_count}, f, ensure_ascii=False, indent=2)


@dataclass
class TokenStats:
    request_tokens: int    # actual tokens sent (from API)
    response_tokens: int   # actual tokens in reply (from API)
    history_tokens: int    # estimated tokens if full history were sent
    summary_tokens: int    # estimated tokens of the summary text
    context_mode: str      # "full" or "compressed"


class SummaryAgent:
    """
    Conversational agent with summary-based context compression.

    History layout in memory:
      [0 .. covered_count-1]  — summarized, stored in summary.json
      [covered_count ..]      — unsummarized; last RECENT_KEEP sent verbatim

    Summarization is triggered whenever the unsummarized segment outside
    the recent window reaches SUMMARY_CHUNK messages.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        history_file: Path,
        summary_file: Path,
        recent_keep: int = RECENT_KEEP,
        summary_chunk: int = SUMMARY_CHUNK,
    ) -> None:
        self._client = client
        self._model = model
        self._system = {"role": "system", "content": system_prompt}
        self._history_file = history_file
        self._summary_file = summary_file
        self._recent_keep = recent_keep
        self._summary_chunk = summary_chunk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_summarize(self) -> None:
        """Summarize old messages if enough have piled up past the recent window."""
        summary = _load_summary(self._summary_file)
        all_msgs = _read_history(self._history_file)
        unsummarized = all_msgs[summary["covered_count"]:]

        # messages outside the recent window that are not yet summarized
        to_summarize = unsummarized[: max(0, len(unsummarized) - self._recent_keep)]

        if len(to_summarize) < self._summary_chunk:
            return  # not enough accumulated yet

        existing = summary["content"]
        chunk_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in to_summarize
        )

        if existing:
            prompt = (
                f"Existing summary:\n{existing}\n\n"
                "Extend the summary by incorporating the new conversation below. "
                "Preserve all specific facts, numbers, names, dates and decisions.\n\n"
                f"New conversation:\n{chunk_text}\n\nUpdated summary:"
            )
        else:
            prompt = (
                "Summarize the conversation below. "
                "Preserve all specific facts, numbers, names, dates and decisions.\n\n"
                f"Conversation:\n{chunk_text}\n\nSummary:"
            )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a precise summarizer. Preserve all specific details."},
                {"role": "user", "content": prompt},
            ],
        )
        new_content = response.choices[0].message.content.strip()
        new_covered = summary["covered_count"] + len(to_summarize)
        _save_summary(self._summary_file, new_content, new_covered)

    def _build_context(self) -> list[dict]:
        """
        Returns the messages list to send:
          system prompt
          + optional summary injection (user/assistant pair)
          + last RECENT_KEEP unsummarized messages
        """
        summary = _load_summary(self._summary_file)
        all_msgs = _read_history(self._history_file)
        recent = all_msgs[summary["covered_count"]:][-self._recent_keep:]

        messages: list[dict] = [self._system]
        if summary["content"]:
            messages.append({
                "role": "user",
                "content": f"[Summary of our earlier conversation:\n{summary['content']}]",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood, I have the context from our earlier conversation.",
            })
        messages.extend(recent)
        return messages

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> tuple[str, TokenStats]:
        user_msg = {"role": "user", "content": user_message}
        _append_messages(self._history_file, user_msg)

        self._maybe_summarize()

        messages = self._build_context()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        reply = response.choices[0].message.content
        _append_messages(self._history_file, {"role": "assistant", "content": reply})

        summary = _load_summary(self._summary_file)
        all_msgs = _read_history(self._history_file)
        stats = TokenStats(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            history_tokens=_estimate_tokens([self._system] + all_msgs),
            summary_tokens=len(summary["content"]) // 4 if summary["content"] else 0,
            context_mode="compressed" if summary["content"] else "full",
        )
        return reply, stats

    def status(self) -> dict:
        summary = _load_summary(self._summary_file)
        all_msgs = _read_history(self._history_file)
        unsummarized = all_msgs[summary["covered_count"]:]
        return {
            "total_messages": len(all_msgs),
            "summarized_messages": summary["covered_count"],
            "unsummarized_messages": len(unsummarized),
            "recent_in_context": min(len(unsummarized), self._recent_keep),
            "has_summary": bool(summary["content"]),
            "summary_tokens": len(summary["content"]) // 4 if summary["content"] else 0,
        }

    def reset(self) -> None:
        self._history_file.unlink(missing_ok=True)
        self._summary_file.unlink(missing_ok=True)


def main() -> None:
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")
    agent = SummaryAgent(client, MODEL, SYSTEM_PROMPT, HISTORY_FILE, SUMMARY_FILE)

    status = agent.status()
    if status["total_messages"]:
        print(
            f"Resuming session — {status['total_messages']} messages total, "
            f"{status['summarized_messages']} summarized"
        )
    else:
        print("New session started.")
    print(f"Model: {MODEL} | keep last {RECENT_KEEP} raw, summarize every {SUMMARY_CHUNK}")
    print("Commands: /status, /reset, /quit\n")

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
            print("--- history and summary cleared ---\n")
            continue
        if user_input == "/status":
            s = agent.status()
            summary_label = (
                f"summary ({s['summary_tokens']} tokens)" if s["has_summary"] else "no summary yet"
            )
            print(
                f"[total: {s['total_messages']} msgs | summarized: {s['summarized_messages']} | "
                f"in context: {s['recent_in_context']} recent + {summary_label}]\n"
            )
            continue

        reply, tok = agent.chat(user_input)
        print(f"Agent: {reply}")
        print(
            f"[tokens — request: {tok.request_tokens} | response: {tok.response_tokens} | "
            f"full history: ~{tok.history_tokens} | mode: {tok.context_mode}]\n"
        )


if __name__ == "__main__":
    main()
