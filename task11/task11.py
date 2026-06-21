"""
Task 11: Three-layer memory for a conversational assistant.

Memory layers (each stored in ONE separate text file)
─────────────────────────────────────────────────────
  1. short-term   — the current dialog (recent turns). Ephemeral, cleared on
                    /reset or a new session.        →  memory/short_term.jsonl
  2. working      — data about the CURRENT task. Cleared with /endtask or /task.
                                                    →  memory/working.txt
  3. long-term    — durable knowledge: profile, decisions, knowledge.
                    Never auto-cleared.             →  memory/long_term.txt

Each fact is one line in its layer's file, so dumping data into a layer is a
single append. Long-term lines carry a [category] tag: profile | decision | knowledge.

What goes where is chosen EXPLICITLY:
  • an LLM "router" reads each user turn and decides which durable facts to keep
    and into which layer/category they belong (or that nothing should be kept);
  • the user can override the router with /remember, /task, /endtask, /forget.

How it affects answers
──────────────────────
  The context sent to the model is assembled as:
      system prompt
    + long-term facts injection      (who the user is, past decisions, knowledge)
    + working facts injection        (the active task)
    + last N short-term dialog turns
  Drop a layer and the assistant loses the corresponding kind of knowledge —
  /status and /facts let you inspect exactly what each layer holds.

Stack: identical to task1 — DeepSeek via the OpenAI client.

Interactive commands
────────────────────
  /memory | /status        show all three layers
  /facts <layer>           list facts in a layer (working | long_term | short_term)
  /remember <layer> <text> manually store a fact
                           layer = working | profile | decision | knowledge
  /task <name>             start a new task (clears working memory)
  /endtask                 finish the task (clears working memory)
  /forget <layer>          wipe a layer (short_term | working | long_term)
  /reset                   clear the current dialog (short-term)
  /quit                    exit (short-term memory is auto-cleared on exit)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

CONFIG_FILE = Path(__file__).parent.parent / "credentials.json"
MEMORY_DIR = Path(__file__).parent / "memory"
MODEL = "deepseek-chat"
RECENT_KEEP = 6  # short-term dialog turns kept verbatim in every request

SYSTEM_PROMPT = (
    "You are an experienced architect and data engineer with 15 years of experience. "
    "Reply concisely, be professional, highlight what is important for data and business. "
    "Use the provided long-term and working memory to personalise and ground your answers."
)

# Long-term facts are organised into these categories.
LONG_TERM_CATEGORIES = ("profile", "decision", "knowledge")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open() as f:
            return json.load(f).get("DEEPSEEK_API_KEY", "")
    return ""


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "") or "") // 4 + 4 for m in messages)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


# ── Layer 1: short-term dialog ───────────────────────────────────────────────────

class ShortTermMemory:
    """The current dialog, stored in one file. Recent turns are replayed to the model."""

    def __init__(self, path: Path, recent_keep: int = RECENT_KEEP) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._recent_keep = recent_keep

    def append(self, role: str, content: str) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")

    def all(self) -> list[dict]:
        msgs: list[dict] = []
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        msgs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return msgs

    def recent(self) -> list[dict]:
        return self.all()[-self._recent_keep:]

    def clear(self) -> None:
        self._path.unlink(missing_ok=True)

    def count(self) -> int:
        return len(self.all())


# ── Layers 2 & 3: one text file per layer, one fact per line ─────────────────────

@dataclass
class Fact:
    content: str
    category: str = ""  # only used by the long-term layer


class FactFile:
    """
    All facts of one layer live in a single text file, one fact per line.

    Long-term lines are tagged with their category, e.g.
        [profile] User's name is Dmitrii.
    Working lines have no tag — they all belong to the current task.

    Appending a fact is a single line write, so anything can be dumped into the
    right layer with one command.
    """

    def __init__(self, path: Path, tagged: bool) -> None:
        self._path = path
        self._tagged = tagged  # True for long-term (category prefixes), False for working
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse(line: str) -> Fact:
        line = line.strip()
        m = re.match(r"^\[([a-z_]+)\]\s*(.*)$", line)
        if m:
            return Fact(content=m.group(2).strip(), category=m.group(1))
        return Fact(content=line, category="")

    def all(self) -> list[Fact]:
        if not self._path.exists():
            return []
        facts = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                facts.append(self._parse(line))
        return facts

    def add(self, content: str, category: str = "") -> bool:
        """Append one fact. Returns False if it's a duplicate (nothing written)."""
        content = content.strip()
        if not content:
            return False
        norm = _normalize(content)
        for existing in self.all():
            en = _normalize(existing.content)
            if norm == en or norm in en or en in norm:
                return False
        if self._tagged and category:
            line = f"[{category}] {content}"
        else:
            line = content
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True

    def clear(self) -> int:
        n = self.count()
        self._path.unlink(missing_ok=True)
        return n

    def count(self) -> int:
        return len(self.all())


# ── The router: decides what to store and where ──────────────────────────────────

_ROUTER_PROMPT = """\
You are the memory router for an assistant. Read the latest USER message and decide \
which durable facts (if any) should be saved, and into which memory layer.

Memory layers:
- "working":   data specific to the CURRENT task only (task goal, scope, current \
numbers/constraints/datasets being worked on right now). Forgotten when the task ends.
- "profile":   durable facts about WHO the user is (name, role, company, team, \
stack they use, stable preferences). [long-term]
- "decision":  decisions, agreements, chosen approaches that should persist. [long-term]
- "knowledge": reusable domain knowledge / general facts worth remembering across tasks. [long-term]

Rules:
- Extract only concrete, atomic facts worth remembering. One fact per item.
- Most chit-chat, questions and acknowledgements produce NO facts — return an empty list.
- Do NOT save the assistant's future answers, only facts stated/implied by the user.
- Keep each fact a short self-contained sentence.

Return ONLY valid JSON, no markdown:
{{"facts": [{{"layer": "profile|decision|knowledge|working", "content": "..."}}]}}

Current task: {task}

USER message:
{message}

JSON:"""


class MemoryRouter:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def route(self, message: str, task: str) -> list[dict]:
        prompt = _ROUTER_PROMPT.format(task=task or "(none)", message=message)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You route facts into memory layers. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            return []
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        facts = data.get("facts", []) if isinstance(data, dict) else []
        valid_layers = {"working", *LONG_TERM_CATEGORIES}
        out = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            layer = f.get("layer", "")
            content = (f.get("content") or "").strip()
            if layer in valid_layers and content:
                out.append({"layer": layer, "content": content})
        return out


@dataclass
class TurnStats:
    request_tokens: int
    response_tokens: int
    history_tokens: int          # cost if the whole dialog were sent raw
    routed: list[dict] = field(default_factory=list)  # facts saved this turn
    layer_counts: dict = field(default_factory=dict)
    interrupted: bool = False    # True if the user aborted generation with Ctrl-C


# ── The agent that ties the three layers together ────────────────────────────────

class MemoryAgent:
    """
    Conversational agent backed by three separate memory layers.

    Per turn:
      1. router classifies the user message → durable facts (working / long-term)
      2. facts are appended to their layer's own file
      3. context = system + long-term + working + recent dialog
      4. model replies; the dialog turn is appended to short-term memory
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        memory_dir: Path = MEMORY_DIR,
        recent_keep: int = RECENT_KEEP,
    ) -> None:
        self._client = client
        self._model = model
        self._system = {"role": "system", "content": system_prompt}
        self._router = MemoryRouter(client, model)

        memory_dir.mkdir(parents=True, exist_ok=True)
        self.short_term = ShortTermMemory(memory_dir / "short_term.jsonl", recent_keep)
        self.working = FactFile(memory_dir / "working.txt", tagged=False)
        self.long_term = FactFile(memory_dir / "long_term.txt", tagged=True)

        self._task_file = memory_dir / "task.txt"
        self._task = self._task_file.read_text(encoding="utf-8").strip() if self._task_file.exists() else ""

    # ── task tracking ────────────────────────────────────────────────────────────

    @property
    def task(self) -> str:
        return self._task

    def start_task(self, name: str) -> str:
        cleared = self.working.clear()
        self._task = name.strip()
        self._task_file.write_text(self._task, encoding="utf-8")
        return f"Started task: '{self._task}' (cleared {cleared} working fact(s))."

    def end_task(self) -> str:
        old = self._task
        cleared = self.working.clear()
        self._task = ""
        self._task_file.unlink(missing_ok=True)
        return f"Ended task '{old or '(none)'}', cleared {cleared} working fact(s)."

    # ── manual fact entry ──────────────────────────────────────────────────────────

    def remember(self, layer: str, content: str) -> str:
        """layer in {working, profile, decision, knowledge}."""
        if layer == "working":
            saved = self.working.add(content)
        elif layer in LONG_TERM_CATEGORIES:
            saved = self.long_term.add(content, category=layer)
        else:
            return f"Unknown layer '{layer}'. Use: working | {' | '.join(LONG_TERM_CATEGORIES)}"
        if not saved:
            return "Already known (duplicate) — nothing saved."
        return f"Saved to {layer}: {content}"

    # ── context assembly ────────────────────────────────────────────────────────────

    def _long_term_block(self) -> str:
        facts = self.long_term.all()
        if not facts:
            return ""
        by_cat: dict[str, list[str]] = {}
        for f in facts:
            by_cat.setdefault(f.category or "knowledge", []).append(f.content)
        lines = []
        for cat in LONG_TERM_CATEGORIES:
            if by_cat.get(cat):
                lines.append(f"{cat.upper()}:")
                lines.extend(f"  - {c}" for c in by_cat[cat])
        return "\n".join(lines)

    def _working_block(self) -> str:
        facts = self.working.all()
        if not facts and not self._task:
            return ""
        lines = []
        if self._task:
            lines.append(f"Current task: {self._task}")
        if facts:
            lines.append("Task data:")
            lines.extend(f"  - {f.content}" for f in facts)
        return "\n".join(lines)

    def _build_context(self) -> list[dict]:
        messages: list[dict] = [self._system]

        long_block = self._long_term_block()
        if long_block:
            messages.append({"role": "user", "content": f"[Long-term memory about me:\n{long_block}]"})
            messages.append({"role": "assistant", "content": "Noted — I'll keep your profile, decisions and knowledge in mind."})

        work_block = self._working_block()
        if work_block:
            messages.append({"role": "user", "content": f"[Working memory (current task):\n{work_block}]"})
            messages.append({"role": "assistant", "content": "Understood, I have the current task context."})

        messages.extend(self.short_term.recent())
        return messages

    # ── main entry point ────────────────────────────────────────────────────────────

    def chat(self, user_message: str, on_token=None) -> tuple[str, TurnStats]:
        """
        Run one turn. If `on_token` is given, the model reply is streamed and each
        text chunk is passed to it as it arrives — this keeps the assistant
        responsive and lets the user abort a long generation with Ctrl-C.

        Ctrl-C is handled gracefully:
          • during fact routing (before the turn is recorded) it re-raises, so the
            whole turn is dropped and the caller returns to the prompt;
          • during generation the partial reply is kept and saved, the turn is
            recorded consistently, and stats.interrupted is set to True.
        """
        # 1. route the message into durable layers (Ctrl-C here aborts the turn)
        routed = self._router.route(user_message, self._task)
        saved: list[dict] = []
        for item in routed:
            layer, content = item["layer"], item["content"]
            if layer == "working":
                ok = self.working.add(content)
            else:
                ok = self.long_term.add(content, category=layer)
            if ok:
                saved.append({"layer": layer, "content": content})

        # 2. short-term: record the user turn
        self.short_term.append("user", user_message)

        # 3. assemble context and call the model
        messages = self._build_context()
        if on_token is not None:
            reply, req_tok, res_tok, interrupted = self._stream(messages, on_token)
        else:
            response = self._client.chat.completions.create(model=self._model, messages=messages)
            reply = response.choices[0].message.content
            req_tok, res_tok, interrupted = (
                response.usage.prompt_tokens, response.usage.completion_tokens, False,
            )

        # 4. short-term: record the reply (even a partial one, to keep the dialog consistent)
        if interrupted and not reply:
            reply = "[interrupted]"
        self.short_term.append("assistant", reply)

        stats = TurnStats(
            request_tokens=req_tok,
            response_tokens=res_tok,
            history_tokens=_estimate_tokens([self._system] + self.short_term.all()),
            routed=saved,
            layer_counts=self.layer_counts(),
            interrupted=interrupted,
        )
        return reply, stats

    def _stream(self, messages: list[dict], on_token) -> tuple[str, int, int, bool]:
        """Stream the reply, forwarding chunks to on_token. Returns (text, req, res, interrupted)."""
        parts: list[str] = []
        req_tok = res_tok = 0
        interrupted = False
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        try:
            for chunk in stream:
                if getattr(chunk, "usage", None):  # final usage-only chunk
                    req_tok = chunk.usage.prompt_tokens
                    res_tok = chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    on_token(delta)
        except KeyboardInterrupt:
            interrupted = True
            stream.close()
        reply = "".join(parts)
        if not res_tok:  # streaming without usage info → estimate
            res_tok = len(reply) // 4
            req_tok = req_tok or _estimate_tokens(messages)
        return reply, req_tok, res_tok, interrupted

    # ── inspection ────────────────────────────────────────────────────────────────

    def layer_counts(self) -> dict:
        return {
            "short_term": self.short_term.count(),
            "working": self.working.count(),
            "long_term": self.long_term.count(),
        }

    def status(self) -> dict:
        long_by_cat: dict[str, int] = {c: 0 for c in LONG_TERM_CATEGORIES}
        for f in self.long_term.all():
            long_by_cat[f.category if f.category in long_by_cat else "knowledge"] += 1
        return {
            "task": self._task or "(none)",
            "short_term_messages": self.short_term.count(),
            "working_facts": self.working.count(),
            "long_term_facts": self.long_term.count(),
            "long_term_by_category": long_by_cat,
        }

    def facts(self, layer: str) -> list[str]:
        if layer == "short_term":
            return [f"{m['role']}: {m['content']}" for m in self.short_term.all()]
        if layer == "working":
            return [f.content for f in self.working.all()]
        if layer == "long_term":
            return [f"[{f.category}] {f.content}" for f in self.long_term.all()]
        return []

    def forget(self, layer: str) -> str:
        if layer == "short_term":
            self.short_term.clear()
            return "Cleared short-term dialog."
        if layer == "working":
            n = self.working.clear()
            return f"Cleared {n} working fact(s)."
        if layer == "long_term":
            n = self.long_term.clear()
            return f"Cleared {n} long-term fact(s)."
        return f"Unknown layer '{layer}'. Use: short_term | working | long_term"


# ── Interactive main ─────────────────────────────────────────────────────────────

def _print_status(agent: MemoryAgent) -> None:
    s = agent.status()
    print(
        f"[task: {s['task']} | short-term: {s['short_term_messages']} msgs | "
        f"working: {s['working_facts']} facts | "
        f"long-term: {s['long_term_facts']} facts {s['long_term_by_category']}]\n"
    )


def main() -> None:
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")
    agent = MemoryAgent(client)

    print(f"Model: {MODEL}  |  three-layer memory (short-term / working / long-term)")
    print("Commands: /memory /facts <layer> /remember <layer> <text> /task <name> /endtask /forget <layer> /reset /quit")
    _print_status(agent)

    try:
        _chat_loop(agent)
    finally:
        # Short-term memory is the current dialog only — wipe it on exit.
        agent.short_term.clear()
        print("(short-term dialog cleared)")


def _chat_loop(agent: MemoryAgent) -> None:
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            print("Bye!")
            return
        if user_input in ("/memory", "/status"):
            _print_status(agent)
            continue
        if user_input == "/reset":
            agent.short_term.clear()
            print("--- short-term dialog cleared ---\n")
            continue
        if user_input.startswith("/facts"):
            layer = user_input[len("/facts"):].strip() or "long_term"
            items = agent.facts(layer)
            if not items:
                print(f"[{layer}] (empty)\n")
            else:
                print(f"[{layer}] {len(items)} item(s):")
                for it in items:
                    print(f"  - {it}")
                print()
            continue
        if user_input.startswith("/remember"):
            rest = user_input[len("/remember"):].strip()
            parts = rest.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /remember <working|profile|decision|knowledge> <text>\n")
                continue
            print(agent.remember(parts[0], parts[1]) + "\n")
            continue
        if user_input.startswith("/task"):
            name = user_input[len("/task"):].strip()
            if not name:
                print("Usage: /task <name>\n")
                continue
            print(agent.start_task(name) + "\n")
            continue
        if user_input == "/endtask":
            print(agent.end_task() + "\n")
            continue
        if user_input.startswith("/forget"):
            layer = user_input[len("/forget"):].strip()
            print(agent.forget(layer) + "\n")
            continue

        reply, stats = agent.chat(user_input)
        print(f"Agent: {reply}")
        if stats.routed:
            saved = "; ".join(f"{r['layer']}: {r['content']}" for r in stats.routed)
            print(f"  ↳ saved → {saved}")
        print(
            f"[tokens — req: {stats.request_tokens} | res: {stats.response_tokens} | "
            f"raw-history: ~{stats.history_tokens} | layers: {stats.layer_counts}]\n"
        )


if __name__ == "__main__":
    main()
