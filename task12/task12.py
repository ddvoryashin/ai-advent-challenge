"""
Task 12: Personalisation on top of the three-layer memory model (task 11).

What's new compared to task 11
──────────────────────────────
  1. USER PROFILE (personalisation).  A structured profile of preferences —
     name, role, style, format, language and constraints — lives in ONE JSON
     file (memory/profile.json) and is a separate entity from the memory layers.
     It is connected to EVERY request automatically: a personalisation block is
     injected at the very front of the context so the assistant honours the
     user's style/format/constraints in every reply without being reminded.

  2. THE TASK LIVES IN THE WORKING LAYER, NOT IN A SEPARATE FILE.
     In task 11 the current task name was persisted to its own memory/task.txt.
     Here there is no task.txt at all: the task is stored as the first fact of
     the WORKING layer (`Current task: <name>`), so it flows into the context
     through the working block and is cleared together with the task.

Memory layers (unchanged from task 11)
──────────────────────────────────────
  1. short-term   — the current dialog (recent turns).   →  memory/short_term.jsonl
  2. working      — data about the CURRENT task + the     →  memory/working.txt
                    task name itself (first line).
  3. long-term    — durable profile/decisions/knowledge.  →  memory/long_term.txt

How a request is assembled
───────────────────────────
      system prompt
    + personalisation block   (profile: style / format / language / constraints)   ← NEW
    + long-term facts
    + working facts (incl. the current task)
    + last N short-term dialog turns

Stack: identical to task1 — DeepSeek via the OpenAI client.

Interactive commands
────────────────────
  /memory | /status                show all layers + active profile
  /facts <layer>                   list facts (working | long_term | short_term)
  /remember <layer> <text>         manually store a fact (working|profile|decision|knowledge)
  /task <name>                     start a task (stored in WORKING memory)
  /endtask                         finish the task (clears WORKING memory)
  /forget <layer>                  wipe a layer (short_term | working | long_term)
  /reset                           clear the current dialog (short-term)
  /profile                         show the active profile
  /profile set <field> <value>     set name|role|style|format|language
  /profile constraint add <text>   add a constraint
  /profile constraint del <n|text> remove a constraint (by index or text)
  /profile preset <name>           load a preset (analyst_ru | casual_en | exec)
  /profile clear                   reset the profile
  /quit                            exit (short-term memory is auto-cleared on exit)
"""

import json
import re
from dataclasses import dataclass, field, asdict
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

# The current task is stored as the first WORKING fact, with this prefix.
TASK_PREFIX = "Current task: "


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
    Working lines have no tag — they all belong to the current task (the task
    name itself is the first working line, prefixed with TASK_PREFIX).
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

    def prepend(self, content: str) -> None:
        """Write `content` as the FIRST line, keeping existing lines after it."""
        content = content.strip()
        if not content:
            return
        existing = [f.content for f in self.all()]
        with self._path.open("w", encoding="utf-8") as f:
            f.write(content + "\n")
            for c in existing:
                f.write(c + "\n")

    def clear(self) -> int:
        n = self.count()
        self._path.unlink(missing_ok=True)
        return n

    def count(self) -> int:
        return len(self.all())


# ── Personalisation: the user profile ────────────────────────────────────────────

@dataclass
class UserProfile:
    """
    Structured personalisation, persisted to a single JSON file. Separate from
    the memory layers — it describes HOW the user wants to be answered, not WHAT
    the assistant knows. Injected into every request via `to_block()`.
    """

    name: str = ""
    role: str = ""
    style: str = ""        # e.g. "concise and professional"
    format: str = ""       # e.g. "short markdown bullet points"
    language: str = ""     # e.g. "Russian"
    constraints: list[str] = field(default_factory=list)

    SETTABLE = ("name", "role", "style", "format", "language")

    @classmethod
    def load(cls, path: Path) -> "UserProfile":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            return cls(
                name=data.get("name", ""),
                role=data.get("role", ""),
                style=data.get("style", ""),
                format=data.get("format", ""),
                language=data.get("language", ""),
                constraints=list(data.get("constraints", [])),
            )
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def is_empty(self) -> bool:
        return not any([self.name, self.role, self.style, self.format, self.language, self.constraints])

    def set_field(self, field_name: str, value: str) -> str:
        field_name = field_name.lower()
        if field_name not in self.SETTABLE:
            return f"Unknown field '{field_name}'. Use: {' | '.join(self.SETTABLE)}"
        setattr(self, field_name, value.strip())
        return f"Set {field_name} = {value.strip()!r}"

    def add_constraint(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Empty constraint — nothing added."
        if text in self.constraints:
            return "Already present — nothing added."
        self.constraints.append(text)
        return f"Added constraint: {text}"

    def del_constraint(self, ref: str) -> str:
        ref = ref.strip()
        if ref.isdigit():
            i = int(ref) - 1
            if 0 <= i < len(self.constraints):
                removed = self.constraints.pop(i)
                return f"Removed constraint: {removed}"
            return f"No constraint #{ref}."
        if ref in self.constraints:
            self.constraints.remove(ref)
            return f"Removed constraint: {ref}"
        return f"No such constraint: {ref}"

    def to_block(self) -> str:
        """Render the profile as a personalisation instruction (empty if no profile)."""
        if self.is_empty():
            return ""
        lines = ["User profile & preferences — always honour these in every reply:"]
        if self.name:
            who = f"Address the user as {self.name}"
            if self.role:
                who += f" (role: {self.role})"
            lines.append(f"- {who}.")
        elif self.role:
            lines.append(f"- The user's role is {self.role}.")
        if self.style:
            lines.append(f"- Style: {self.style}.")
        if self.format:
            lines.append(f"- Format: {self.format}.")
        if self.language:
            lines.append(f"- Always respond in {self.language}.")
        if self.constraints:
            lines.append("- Constraints:")
            lines.extend(f"    - {c}" for c in self.constraints)
        return "\n".join(lines)

    def summary(self) -> str:
        if self.is_empty():
            return "(no profile set)"
        parts = []
        if self.name:
            parts.append(f"name={self.name}")
        if self.role:
            parts.append(f"role={self.role}")
        if self.style:
            parts.append(f"style={self.style}")
        if self.format:
            parts.append(f"format={self.format}")
        if self.language:
            parts.append(f"language={self.language}")
        if self.constraints:
            parts.append(f"constraints={self.constraints}")
        return " | ".join(parts)


# A few ready-made profiles to make "answers for different profiles" easy to try.
PROFILE_PRESETS: dict[str, dict] = {
    "analyst_ru": {
        "name": "",
        "role": "data analyst",
        "style": "concise, professional, data-driven",
        "format": "short markdown bullet points",
        "language": "Russian",
        "constraints": ["avoid marketing fluff", "include concrete numbers where possible"],
    },
    "casual_en": {
        "name": "",
        "role": "",
        "style": "friendly, conversational, encouraging",
        "format": "plain prose paragraphs, no bullet lists",
        "language": "English",
        "constraints": ["keep it under 120 words", "no jargon"],
    },
    "exec": {
        "name": "",
        "role": "executive sponsor",
        "style": "high-level, business-focused",
        "format": "a 3-bullet executive summary",
        "language": "English",
        "constraints": ["no code", "focus on business impact and risk, not implementation"],
    },
}


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


# ── The agent that ties the three layers + the profile together ──────────────────

class MemoryAgent:
    """
    Conversational agent backed by three memory layers and a user profile.

    Per turn:
      1. router classifies the user message → durable facts (working / long-term)
      2. facts are appended to their layer's own file
      3. context = system + PROFILE + long-term + working (incl. task) + recent dialog
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

        # Personalisation profile (separate JSON entity, no task.txt anywhere).
        self._profile_path = memory_dir / "profile.json"
        self.profile = UserProfile.load(self._profile_path)

    # ── profile management ─────────────────────────────────────────────────────────

    def _save_profile(self) -> None:
        self.profile.save(self._profile_path)

    def set_profile_field(self, field_name: str, value: str) -> str:
        msg = self.profile.set_field(field_name, value)
        self._save_profile()
        return msg

    def add_constraint(self, text: str) -> str:
        msg = self.profile.add_constraint(text)
        self._save_profile()
        return msg

    def del_constraint(self, ref: str) -> str:
        msg = self.profile.del_constraint(ref)
        self._save_profile()
        return msg

    def load_preset(self, name: str) -> str:
        preset = PROFILE_PRESETS.get(name)
        if not preset:
            return f"Unknown preset '{name}'. Available: {', '.join(PROFILE_PRESETS)}"
        self.profile = UserProfile(
            name=preset.get("name", ""),
            role=preset.get("role", ""),
            style=preset.get("style", ""),
            format=preset.get("format", ""),
            language=preset.get("language", ""),
            constraints=list(preset.get("constraints", [])),
        )
        self._save_profile()
        return f"Loaded preset '{name}': {self.profile.summary()}"

    def clear_profile(self) -> str:
        self.profile = UserProfile()
        self._profile_path.unlink(missing_ok=True)
        return "Profile cleared."

    # ── task tracking (stored in the WORKING layer, no separate file) ───────────────

    @property
    def task(self) -> str:
        for f in self.working.all():
            if f.content.startswith(TASK_PREFIX):
                return f.content[len(TASK_PREFIX):].strip()
        return ""

    def start_task(self, name: str) -> str:
        cleared = self.working.clear()
        name = name.strip()
        self.working.prepend(f"{TASK_PREFIX}{name}")
        return f"Started task: '{name}' (cleared {cleared} working fact(s))."

    def end_task(self) -> str:
        old = self.task
        cleared = self.working.clear()
        return f"Ended task '{old or '(none)'}', cleared {cleared} working item(s)."

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
        if not facts:
            return ""
        task = self.task
        data = [f.content for f in facts if not f.content.startswith(TASK_PREFIX)]
        lines = []
        if task:
            lines.append(f"Current task: {task}")
        if data:
            lines.append("Task data:")
            lines.extend(f"  - {c}" for c in data)
        return "\n".join(lines)

    def _build_context(self) -> list[dict]:
        messages: list[dict] = [self._system]

        # Personalisation is connected to EVERY request, right after the system prompt.
        profile_block = self.profile.to_block()
        if profile_block:
            messages.append({"role": "system", "content": profile_block})

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
        """
        # 1. route the message into durable layers
        routed = self._router.route(user_message, self.task)
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

        # 3. assemble context (system + profile + memory + dialog) and call the model
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
            "working": self._working_data_count(),
            "long_term": self.long_term.count(),
        }

    def _working_data_count(self) -> int:
        """Working facts excluding the task marker line."""
        return sum(1 for f in self.working.all() if not f.content.startswith(TASK_PREFIX))

    def status(self) -> dict:
        long_by_cat: dict[str, int] = {c: 0 for c in LONG_TERM_CATEGORIES}
        for f in self.long_term.all():
            long_by_cat[f.category if f.category in long_by_cat else "knowledge"] += 1
        return {
            "task": self.task or "(none)",
            "short_term_messages": self.short_term.count(),
            "working_facts": self._working_data_count(),
            "long_term_facts": self.long_term.count(),
            "long_term_by_category": long_by_cat,
            "profile": self.profile.summary(),
        }

    def facts(self, layer: str) -> list[str]:
        if layer == "short_term":
            return [f"{m['role']}: {m['content']}" for m in self.short_term.all()]
        if layer == "working":
            return [f.content for f in self.working.all() if not f.content.startswith(TASK_PREFIX)]
        if layer == "long_term":
            return [f"[{f.category}] {f.content}" for f in self.long_term.all()]
        return []

    def forget(self, layer: str) -> str:
        if layer == "short_term":
            self.short_term.clear()
            return "Cleared short-term dialog."
        if layer == "working":
            n = self.working.clear()
            return f"Cleared {n} working item(s)."
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
        f"long-term: {s['long_term_facts']} facts {s['long_term_by_category']}]"
    )
    print(f"[profile: {s['profile']}]\n")


def _handle_profile_command(agent: MemoryAgent, rest: str) -> None:
    rest = rest.strip()
    if not rest or rest == "show":
        block = agent.profile.to_block()
        print(f"[profile] {agent.profile.summary()}")
        if block:
            print("--- injected into every request ---")
            print(block)
        print()
        return
    parts = rest.split(maxsplit=1)
    sub = parts[0]
    arg = parts[1] if len(parts) > 1 else ""
    if sub == "set":
        kv = arg.split(maxsplit=1)
        if len(kv) < 2:
            print("Usage: /profile set <name|role|style|format|language> <value>\n")
            return
        print(agent.set_profile_field(kv[0], kv[1]) + "\n")
    elif sub == "constraint":
        cparts = arg.split(maxsplit=1)
        if len(cparts) < 2 or cparts[0] not in ("add", "del"):
            print("Usage: /profile constraint add <text> | /profile constraint del <n|text>\n")
            return
        if cparts[0] == "add":
            print(agent.add_constraint(cparts[1]) + "\n")
        else:
            print(agent.del_constraint(cparts[1]) + "\n")
    elif sub == "preset":
        print(agent.load_preset(arg.strip()) + "\n")
    elif sub == "clear":
        print(agent.clear_profile() + "\n")
    else:
        print(f"Unknown /profile subcommand '{sub}'. Use: set | constraint | preset | clear\n")


def main() -> None:
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")
    agent = MemoryAgent(client)

    print(f"Model: {MODEL}  |  three-layer memory + personalised profile")
    print("Commands: /memory /facts <layer> /remember <layer> <text> /task <name> /endtask")
    print("          /forget <layer> /reset /profile [set|constraint|preset|clear] /quit")
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
        if user_input.startswith("/profile"):
            _handle_profile_command(agent, user_input[len("/profile"):])
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
