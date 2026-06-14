"""
Task 10: Three context management strategies with a unified switcher.

Strategies
──────────
  1. sliding_window  — keep only the last N messages, discard the rest
  2. sticky_facts    — LLM-extracted key-value facts + last N messages
  3. branching       — fork conversations into independent branches

Usage (interactive)
───────────────────
  /strategy <name>   — switch strategy
  /status            — show current strategy state
  /reset             — reset current strategy history
  /quit              — exit

Branching-specific commands (when strategy = branching)
  /branch <name>     — fork from current position into a new branch
  /branches          — list all branches
  /switch <name>     — switch to another branch
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

CONFIG_FILE = Path(__file__).parent.parent / "credentials.json"
DATA_DIR = Path(__file__).parent / "data"
MODEL = "deepseek-chat"
SYSTEM_PROMPT = (
    "You are an experienced architect and data engineer with 15 years of experience. "
    "Reply concisely, be professional, highlight what is important for data and business. "
    "Think about technical and business context of your answers."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "") or "") // 4 + 4 for m in messages)


@dataclass
class TokenStats:
    request_tokens: int
    response_tokens: int
    history_tokens: int
    strategy: str
    extra: dict = field(default_factory=dict)


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    def __init__(self, client: OpenAI, model: str, system_prompt: str) -> None:
        self._client = client
        self._model = model
        self._system = {"role": "system", "content": system_prompt}

    def _call_api(self, messages: list[dict]) -> tuple[str, int, int]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def chat(self, user_message: str) -> tuple[str, TokenStats]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def status(self) -> dict: ...


# ── Strategy 1: Sliding Window ────────────────────────────────────────────────

class SlidingWindowStrategy(BaseStrategy):
    """Keep only the last N messages. Everything older is silently dropped."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        window_size: int = 10,
        storage: Optional[Path] = None,
    ) -> None:
        super().__init__(client, model, system_prompt)
        self._window_size = window_size
        self._storage = storage
        self._history: list[dict] = []
        if storage and storage.exists():
            with storage.open() as f:
                self._history = json.load(f)

    @property
    def name(self) -> str:
        return "sliding_window"

    def _save(self) -> None:
        if self._storage:
            self._storage.parent.mkdir(parents=True, exist_ok=True)
            with self._storage.open("w", encoding="utf-8") as f:
                json.dump(self._history, f, ensure_ascii=False, indent=2)

    def chat(self, user_message: str) -> tuple[str, TokenStats]:
        self._history.append({"role": "user", "content": user_message})
        window = self._history[-self._window_size:]
        messages = [self._system] + window
        reply, req_tok, res_tok = self._call_api(messages)
        self._history.append({"role": "assistant", "content": reply})
        self._save()
        return reply, TokenStats(
            request_tokens=req_tok,
            response_tokens=res_tok,
            history_tokens=_estimate_tokens([self._system] + self._history),
            strategy=self.name,
            extra={
                "window_size": self._window_size,
                "total_messages": len(self._history),
                "dropped": max(0, len(self._history) - self._window_size),
            },
        )

    def reset(self) -> None:
        self._history.clear()
        if self._storage:
            self._storage.unlink(missing_ok=True)

    def status(self) -> dict:
        dropped = max(0, len(self._history) - self._window_size)
        return {
            "strategy": self.name,
            "total_messages": len(self._history),
            "window_size": self._window_size,
            "in_context": min(len(self._history), self._window_size),
            "dropped": dropped,
        }


# ── Strategy 2: Sticky Facts ──────────────────────────────────────────────────

_FACTS_PROMPT = """\
You are a fact extractor for a conversation. Given the existing facts and a new user message,
return an UPDATED facts JSON object.

Rules:
- Keep all existing facts unless explicitly contradicted.
- Add new facts for: goals, constraints, technologies, people, numbers, decisions, SLAs, deadlines.
- Use short, precise keys (snake_case). Values should be concise strings.
- Return ONLY a valid JSON object, no markdown, no explanations.

Existing facts:
{existing}

New user message:
{message}

Updated JSON:"""


class StickyFactsStrategy(BaseStrategy):
    """
    Maintains an LLM-updated key-value facts block.
    Context sent = system + facts injection + last N messages.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        window_size: int = 6,
        storage: Optional[Path] = None,
    ) -> None:
        super().__init__(client, model, system_prompt)
        self._window_size = window_size
        self._storage = storage
        self._history: list[dict] = []
        self._facts: dict[str, str] = {}
        if storage and storage.exists():
            with storage.open() as f:
                data = json.load(f)
                self._history = data.get("history", [])
                self._facts = data.get("facts", {})

    @property
    def name(self) -> str:
        return "sticky_facts"

    def _save(self) -> None:
        if self._storage:
            self._storage.parent.mkdir(parents=True, exist_ok=True)
            with self._storage.open("w", encoding="utf-8") as f:
                json.dump(
                    {"history": self._history, "facts": self._facts},
                    f, ensure_ascii=False, indent=2,
                )

    def _update_facts(self, user_message: str) -> None:
        existing = json.dumps(self._facts, ensure_ascii=False) if self._facts else "{}"
        prompt = _FACTS_PROMPT.format(existing=existing, message=user_message)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "Extract facts. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        # Strip possible markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        try:
            updated = json.loads(raw)
            if isinstance(updated, dict):
                self._facts = updated
        except json.JSONDecodeError:
            pass  # keep existing facts on parse failure

    def _build_context(self) -> list[dict]:
        messages: list[dict] = [self._system]
        if self._facts:
            facts_text = "\n".join(f"  {k}: {v}" for k, v in self._facts.items())
            messages.append({
                "role": "user",
                "content": f"[Conversation facts:\n{facts_text}]",
            })
            messages.append({
                "role": "assistant",
                "content": "Acknowledged. I have the key facts from our conversation.",
            })
        messages.extend(self._history[-self._window_size:])
        return messages

    def chat(self, user_message: str) -> tuple[str, TokenStats]:
        self._update_facts(user_message)
        self._history.append({"role": "user", "content": user_message})
        messages = self._build_context()
        reply, req_tok, res_tok = self._call_api(messages)
        self._history.append({"role": "assistant", "content": reply})
        self._save()
        return reply, TokenStats(
            request_tokens=req_tok,
            response_tokens=res_tok,
            history_tokens=_estimate_tokens([self._system] + self._history),
            strategy=self.name,
            extra={
                "facts_count": len(self._facts),
                "window_size": self._window_size,
                "total_messages": len(self._history),
            },
        )

    def get_facts(self) -> dict[str, str]:
        return dict(self._facts)

    def reset(self) -> None:
        self._history.clear()
        self._facts.clear()
        if self._storage:
            self._storage.unlink(missing_ok=True)

    def status(self) -> dict:
        return {
            "strategy": self.name,
            "total_messages": len(self._history),
            "window_size": self._window_size,
            "in_context": min(len(self._history), self._window_size),
            "facts_count": len(self._facts),
            "facts": self._facts,
        }


# ── Strategy 3: Branching ─────────────────────────────────────────────────────

@dataclass
class Branch:
    name: str
    messages: list[dict]
    parent: Optional[str]
    checkpoint_idx: int  # number of parent messages at branch creation


class BranchingStrategy(BaseStrategy):
    """
    Fork conversations into independent branches.

    Each branch starts as a copy of parent messages up to the fork point,
    then evolves independently. Only the current branch's window is sent to LLM.

    User commands (handled inside chat()):
      /branch <name>  — fork current branch into a new one
      /branches       — list all branches
      /switch <name>  — switch active branch
    """

    _TOPIC_SHIFT_PROMPT = (
        "Two consecutive user messages are shown. "
        "Is the second a *significant topic shift* from the first (different domain/concern)? "
        "Reply with a single word: yes or no."
    )

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        window_size: int = 8,
        storage: Optional[Path] = None,
        auto_branch: bool = False,
    ) -> None:
        super().__init__(client, model, system_prompt)
        self._window_size = window_size
        self._storage = storage
        self._auto_branch = auto_branch
        self._branches: dict[str, Branch] = {
            "main": Branch(name="main", messages=[], parent=None, checkpoint_idx=0)
        }
        self._current = "main"
        if storage and storage.exists():
            self._load()

    @property
    def name(self) -> str:
        return "branching"

    @property
    def current_branch(self) -> Branch:
        return self._branches[self._current]

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        with self._storage.open() as f:
            data = json.load(f)
        self._current = data["current"]
        self._branches = {
            n: Branch(
                name=n,
                messages=b["messages"],
                parent=b["parent"],
                checkpoint_idx=b["checkpoint_idx"],
            )
            for n, b in data["branches"].items()
        }

    def _save(self) -> None:
        if not self._storage:
            return
        self._storage.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current": self._current,
            "branches": {
                n: {
                    "messages": b.messages,
                    "parent": b.parent,
                    "checkpoint_idx": b.checkpoint_idx,
                }
                for n, b in self._branches.items()
            },
        }
        with self._storage.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Branch operations ─────────────────────────────────────────────────────

    def create_branch(self, branch_name: str) -> str:
        if branch_name in self._branches:
            return f"Branch '{branch_name}' already exists."
        parent = self.current_branch
        new_branch = Branch(
            name=branch_name,
            messages=list(parent.messages),
            parent=self._current,
            checkpoint_idx=len(parent.messages),
        )
        self._branches[branch_name] = new_branch
        self._current = branch_name
        self._save()
        return (
            f"Created branch '{branch_name}' from '{parent.name}' "
            f"at message #{len(parent.messages)}. Now on '{branch_name}'."
        )

    def switch_branch(self, branch_name: str) -> str:
        if branch_name not in self._branches:
            available = ", ".join(self._branches)
            return f"Branch '{branch_name}' not found. Available: {available}"
        self._current = branch_name
        self._save()
        b = self._branches[branch_name]
        parent_info = f", forked from '{b.parent}' @{b.checkpoint_idx}" if b.parent else ""
        return (
            f"Switched to '{branch_name}' "
            f"({len(b.messages)} messages{parent_info})."
        )

    def list_branches(self) -> str:
        lines = ["Branches:"]
        for bname, b in self._branches.items():
            marker = " *" if bname == self._current else ""
            parent_info = f" ← '{b.parent}'@{b.checkpoint_idx}" if b.parent else ""
            lines.append(f"  {bname}{parent_info} [{len(b.messages)} msgs]{marker}")
        return "\n".join(lines)

    def _detect_topic_shift(self, new_message: str) -> bool:
        msgs = self.current_branch.messages
        last_user = next(
            (m["content"] for m in reversed(msgs) if m["role"] == "user"), None
        )
        if not last_user:
            return False
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._TOPIC_SHIFT_PROMPT},
                {"role": "user", "content": f"1: {last_user[:400]}\n2: {new_message[:400]}"},
            ],
        )
        return response.choices[0].message.content.strip().lower().startswith("yes")

    # ── Core ──────────────────────────────────────────────────────────────────

    def _build_context(self) -> list[dict]:
        window = self.current_branch.messages[-self._window_size:]
        return [self._system] + window

    def chat(self, user_message: str) -> tuple[str, TokenStats]:
        stripped = user_message.strip()
        null_stats = TokenStats(0, 0, 0, self.name)

        if stripped.startswith("/branch "):
            return self.create_branch(stripped[8:].strip()), null_stats
        if stripped == "/branches":
            return self.list_branches(), null_stats
        if stripped.startswith("/switch "):
            return self.switch_branch(stripped[8:].strip()), null_stats

        if self._auto_branch and len(self.current_branch.messages) >= 2:
            if self._detect_topic_shift(user_message):
                count = len(self._branches)
                self.create_branch(f"branch_{count}")

        branch = self.current_branch
        branch.messages.append({"role": "user", "content": user_message})
        messages = self._build_context()
        reply, req_tok, res_tok = self._call_api(messages)
        branch.messages.append({"role": "assistant", "content": reply})
        self._save()
        return reply, TokenStats(
            request_tokens=req_tok,
            response_tokens=res_tok,
            history_tokens=_estimate_tokens([self._system] + branch.messages),
            strategy=self.name,
            extra={
                "current_branch": self._current,
                "branch_messages": len(branch.messages),
                "total_branches": len(self._branches),
                "window_size": self._window_size,
            },
        )

    def reset(self) -> None:
        self._branches = {"main": Branch(name="main", messages=[], parent=None, checkpoint_idx=0)}
        self._current = "main"
        if self._storage:
            self._storage.unlink(missing_ok=True)

    def status(self) -> dict:
        branch = self.current_branch
        return {
            "strategy": self.name,
            "current_branch": self._current,
            "all_branches": {
                n: {"messages": len(b.messages), "parent": b.parent}
                for n, b in self._branches.items()
            },
            "branch_messages": len(branch.messages),
            "window_size": self._window_size,
        }


# ── Multi-strategy agent ──────────────────────────────────────────────────────

class ContextAgent:
    """
    Unified agent that hosts all three strategies and lets you switch between them.

    Top-level commands:
      /strategy <name>  — switch to sliding_window | sticky_facts | branching
      /status           — show current strategy state
      /reset            — reset current strategy
      /quit             — exit (for interactive use)
    """

    STRATEGIES = ("sliding_window", "sticky_facts", "branching")

    def __init__(
        self,
        client: OpenAI,
        model: str = MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        data_dir: Path = DATA_DIR,
        window_size: int = 10,
    ) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._current_strategy = "sliding_window"
        self._strategies: dict[str, BaseStrategy] = {
            "sliding_window": SlidingWindowStrategy(
                client, model, system_prompt,
                window_size=window_size,
                storage=data_dir / "sw_history.json",
            ),
            "sticky_facts": StickyFactsStrategy(
                client, model, system_prompt,
                window_size=max(4, window_size // 2),
                storage=data_dir / "sf_data.json",
            ),
            "branching": BranchingStrategy(
                client, model, system_prompt,
                window_size=window_size,
                storage=data_dir / "br_data.json",
            ),
        }

    @property
    def strategy(self) -> BaseStrategy:
        return self._strategies[self._current_strategy]

    def switch_strategy(self, name: str) -> str:
        if name not in self._strategies:
            return f"Unknown strategy '{name}'. Available: {', '.join(self.STRATEGIES)}"
        self._current_strategy = name
        return f"Switched to strategy: {name}"

    def get_strategy(self, name: str) -> BaseStrategy:
        return self._strategies[name]

    def chat(self, user_message: str) -> tuple[str, TokenStats]:
        stripped = user_message.strip()
        null_stats = TokenStats(0, 0, 0, self._current_strategy)
        if stripped.startswith("/strategy "):
            return self.switch_strategy(stripped[10:].strip()), null_stats
        if stripped == "/status":
            return str(self.strategy.status()), null_stats
        if stripped == "/reset":
            self.strategy.reset()
            return f"[{self._current_strategy}] Reset.", null_stats
        return self.strategy.chat(user_message)


# ── Interactive main ──────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")
    agent = ContextAgent(client)

    print(f"Model: {MODEL}")
    print("Strategies: sliding_window | sticky_facts | branching")
    print("Commands: /strategy <name>, /status, /reset, /quit")
    print(f"Active: {agent._current_strategy}\n")

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

        reply, tok = agent.chat(user_input)
        print(f"Agent: {reply}")
        if tok.request_tokens:
            print(
                f"[tokens — req: {tok.request_tokens} | res: {tok.response_tokens} | "
                f"hist: ~{tok.history_tokens} | strategy: {tok.strategy} | {tok.extra}]\n"
            )
        else:
            print()


if __name__ == "__main__":
    main()
