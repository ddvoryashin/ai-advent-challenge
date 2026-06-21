"""
Task 13: The task modelled as a finite state machine (FSM), on top of the
personalised three-layer memory of task 12.

What's new compared to task 12
──────────────────────────────
  THE TASK NOW HAS A LIFECYCLE, modelled as an explicit finite state machine.
  A new entity — `TaskState` — tracks, for the current task:

      • stage            — the task's lifecycle stage, ONE OF (exactly):
                               planning → execution → validation → done
      • current step     — the concrete step being worked on inside that stage
      • expected action  — what is expected to happen next

  The FSM can be PAUSED at any stage and RESUMED later. Its full state
  (stage, step, expected action, paused flag, and a per-stage progress log)
  is persisted to ONE JSON file (memory/task_state.json) and reloaded on
  startup — so after a pause (or even closing the program), work CONTINUES
  from exactly where it stopped, WITHOUT the user having to re-explain anything.

  This mirrors how task 12 added the profile as a separate JSON entity: the
  FSM is a separate "task lifecycle" entity, connected to EVERY request via a
  state block injected into the context.

Inherited from task 12 (unchanged)
───────────────────────────────────
  • USER PROFILE (personalisation) in memory/profile.json, injected into every
    request.
  • Three memory layers: short-term dialog, working (task data + task name),
    long-term (profile / decision / knowledge).
  • The task NAME still lives as the first fact of the WORKING layer.

How a request is assembled
───────────────────────────
      system prompt
    + personalisation block      (profile: style / format / language / constraints)
    + TASK STATE block           (FSM: stage / step / expected action / progress)  ← NEW
    + long-term facts
    + working facts (task data)
    + last N short-term dialog turns

Stack: identical to task1 — DeepSeek via the OpenAI client.

Interactive commands
────────────────────
  /memory | /status               show all layers + profile + task state
  /state | /fsm                   show the task state machine in detail
  /facts <layer>                  list facts (working | long_term | short_term)
  /remember <layer> <text>        manually store a fact (working|profile|decision|knowledge)

  /task <name>                    start a task (FSM begins at 'planning'; a plan is drafted)
  /advance | /next                advance the FSM to the next stage
  /back | /prev                   go back to the previous stage
  /step <text>                    set the current step
  /expect <text>                  override the expected action
  /note <text>                    add a progress note (kept across pauses)
  /pause                          HARD-pause: the agent refuses to work until /resume
  /resume                         resume a paused task
  (Ctrl-C while the agent is replying interrupts it and auto-pauses the task)
  /endtask                        finish the task (clears working + resets the FSM)

  /forget <layer>                 wipe a layer (short_term | working | long_term)
  /reset                          clear the current dialog (short-term)
  /profile ...                    manage the personalisation profile
  /quit                           exit (short-term memory is auto-cleared on exit)
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
    "Use the provided long-term and working memory to personalise and ground your answers. "
    "A task-state machine tells you the current stage, step and expected action — always "
    "continue from there and never re-ask the user about things already decided. "
    "IMPORTANT: you are an advisor — you CANNOT run code, execute SQL or touch real "
    "systems. Never claim a step is done or a result achieved unless the user has actually "
    "reported the real outcome (recorded in working memory or progress notes). Clearly "
    "separate what you PROPOSE from what has actually happened, and never fabricate results."
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


# ── Task lifecycle: a finite state machine ───────────────────────────────────────

# The task moves through EXACTLY these stages, in this order.
STAGES = ("planning", "execution", "validation", "done")

# The default "expected action" for each stage — what is expected to happen next.
STAGE_EXPECTED_ACTION = {
    "planning":   "Clarify the goal and produce a concrete, step-by-step plan.",
    "execution":  "Carry out the agreed plan, one step at a time.",
    "validation": "Check the result against the plan and acceptance criteria.",
    "done":       "Task complete — nothing further is expected.",
}

# When the FSM ENTERS one of these stages, the agent immediately does the stage's
# work via the model. ('planning' is handled at /task by plan_task(); 'done' is a
# terminal stage with no action.)
STAGE_WORK_PROMPT = {
    "execution": (
        "We have just entered the EXECUTION stage and the plan from the planning notes "
        "is approved. You CANNOT execute anything yourself, so for each plan step produce "
        "the exact, ready-to-run artifact the user must run (SQL / commands / config) and "
        "mark its status as TODO — not done. Do NOT narrate fictional success and do NOT "
        "use past tense as if a step were already completed. Finish by asking the user to "
        "run the steps and paste back the REAL outputs so they can be validated."
    ),
    "validation": (
        "We have just entered the VALIDATION stage. Validate STRICTLY against evidence "
        "actually recorded in the working memory and progress notes — i.e. real outputs "
        "the user has reported. For every acceptance criterion give a verdict: PASS only "
        "if there is recorded evidence for it; otherwise 'NOT VERIFIED — no result "
        "recorded'. Never assume or invent that a step ran or succeeded. If there is no "
        "real execution evidence, state plainly that the task is NOT yet validated and "
        "list exactly what actual results/evidence you need to proceed."
    ),
}


@dataclass
class TaskState:
    """
    The current task modelled as a finite state machine.

    The three observable parts required by the task:
      • stage           — the lifecycle stage, one of STAGES
      • step            — the current step inside that stage (free text)
      • expected_action — what is expected to happen next

    Plus the bits that make pause / resume work:
      • paused          — the FSM can be paused at ANY stage and resumed later
      • notes           — a per-stage progress log, so work CONTINUES after a
                          pause without the user re-explaining anything

    Persisted to one JSON file and reloaded on startup → seamless resume.
    """

    name: str = ""
    stage: str = "planning"
    step: str = ""
    expected_action: str = ""
    paused: bool = False
    notes: list[str] = field(default_factory=list)  # entries like "planning: ..."

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            self.stage = "planning"
        if not self.expected_action:
            self.expected_action = STAGE_EXPECTED_ACTION.get(self.stage, "")

    # ── queries ─────────────────────────────────────────────────────────────────
    def is_active(self) -> bool:
        return bool(self.name)

    def is_done(self) -> bool:
        return self.stage == "done"

    # ── transitions ──────────────────────────────────────────────────────────────
    def advance(self) -> tuple[bool, str]:
        """Move forward one stage along planning → execution → validation → done."""
        if not self.is_active():
            return False, "No active task. Start one with /task <name>."
        idx = STAGES.index(self.stage)
        if idx >= len(STAGES) - 1:
            return False, f"Already at the final stage '{self.stage}'. Use /endtask to finish."
        old = self.stage
        # Preserve what was being done as a progress note before moving on.
        if self.step:
            self.notes.append(f"{old}: {self.step}")
        self.stage = STAGES[idx + 1]
        self.step = ""
        self.expected_action = STAGE_EXPECTED_ACTION[self.stage]
        return True, f"Stage: {old} → {self.stage}. Expected action: {self.expected_action}"

    def back(self) -> tuple[bool, str]:
        """Move back one stage along done → validation → execution → planning."""
        if not self.is_active():
            return False, "No active task. Start one with /task <name>."
        idx = STAGES.index(self.stage)
        if idx == 0:
            return False, f"Already at the first stage '{self.stage}'."
        old = self.stage
        # Preserve what was being done as a progress note before stepping back.
        if self.step:
            self.notes.append(f"{old}: {self.step}")
        self.stage = STAGES[idx - 1]
        self.step = ""
        self.expected_action = STAGE_EXPECTED_ACTION[self.stage]
        return True, f"Stage: {old} → {self.stage} (went back). Expected action: {self.expected_action}"

    def pause(self) -> str:
        if not self.is_active():
            return "No active task to pause."
        if self.paused:
            return f"Task '{self.name}' is already paused at stage '{self.stage}'."
        self.paused = True
        return (f"Paused at stage '{self.stage}' (step: {self.step or '—'}). "
                f"Resume anytime with /resume — state is saved.")

    def resume(self) -> str:
        if not self.is_active():
            return "No active task to resume."
        if not self.paused:
            return f"Task '{self.name}' is already running (stage '{self.stage}')."
        self.paused = False
        return (f"Resumed task '{self.name}' at stage '{self.stage}'. "
                f"Continuing from: {self.step or self.expected_action}")

    def set_step(self, text: str) -> str:
        if not self.is_active():
            return "No active task. Start one with /task <name>."
        self.step = text.strip()
        return f"Current step set: {self.step or '(cleared)'}"

    def set_expected(self, text: str) -> str:
        if not self.is_active():
            return "No active task. Start one with /task <name>."
        self.expected_action = text.strip() or STAGE_EXPECTED_ACTION[self.stage]
        return f"Expected action set: {self.expected_action}"

    def add_note(self, text: str) -> str:
        if not self.is_active():
            return "No active task. Start one with /task <name>."
        text = text.strip()
        if not text:
            return "Empty note — nothing added."
        self.notes.append(f"{self.stage}: {text}")
        return f"Noted ({self.stage}): {text}"

    # ── rendering ────────────────────────────────────────────────────────────────
    def stage_track(self) -> str:
        return " → ".join(f"[{s}]" if s == self.stage else s for s in STAGES)

    def to_block(self) -> str:
        """The state block injected into every request — the key to seamless resume."""
        if not self.is_active():
            return ""
        lines = [
            "Current task state machine — CONTINUE from exactly here; do NOT re-ask "
            "for anything already decided below:",
            f"- Task: {self.name}",
            f"- Stage: {self.stage}   ({self.stage_track()})",
            f"- Current step: {self.step or '(not set yet)'}",
            f"- Expected action: {self.expected_action}",
            f"- Status: {'PAUSED — hold and wait for /resume' if self.paused else 'active'}",
        ]
        if self.notes:
            lines.append("- Progress so far:")
            lines.extend(f"    - {n}" for n in self.notes)
        return "\n".join(lines)

    def summary(self) -> str:
        if not self.is_active():
            return "(no active task)"
        status = "PAUSED" if self.paused else "active"
        return (f"{self.name} | stage={self.stage} | step={self.step or '—'} | "
                f"expects={self.expected_action} | {status}")

    # ── persistence ──────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: Path) -> "TaskState":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            return cls(
                name=data.get("name", ""),
                stage=data.get("stage", "planning") or "planning",
                step=data.get("step", ""),
                expected_action=data.get("expected_action", ""),
                paused=bool(data.get("paused", False)),
                notes=list(data.get("notes", [])),
            )
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


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


# ── The agent that ties the three layers + the profile + the FSM together ─────────

class MemoryAgent:
    """
    Conversational agent backed by three memory layers, a user profile and a
    task-lifecycle finite state machine.

    Per turn:
      1. router classifies the user message → durable facts (working / long-term)
      2. facts are appended to their layer's own file
      3. context = system + PROFILE + TASK STATE (FSM) + long-term + working + recent dialog
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

        # Personalisation profile (separate JSON entity, from task 12).
        self._profile_path = memory_dir / "profile.json"
        self.profile = UserProfile.load(self._profile_path)

        # Task lifecycle FSM (separate JSON entity, NEW in task 13).
        self._task_state_path = memory_dir / "task_state.json"
        self.task_state = TaskState.load(self._task_state_path)

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

    # ── task tracking: name in WORKING layer, lifecycle in the FSM ──────────────────

    @property
    def task(self) -> str:
        for f in self.working.all():
            if f.content.startswith(TASK_PREFIX):
                return f.content[len(TASK_PREFIX):].strip()
        return ""

    def _save_task_state(self) -> None:
        self.task_state.save(self._task_state_path)

    def start_task(self, name: str) -> str:
        cleared = self.working.clear()
        name = name.strip()
        self.working.prepend(f"{TASK_PREFIX}{name}")
        # The FSM always begins at the first stage, 'planning'.
        self.task_state = TaskState(name=name)
        self._save_task_state()
        return (f"Started task: '{name}' at stage 'planning' "
                f"(cleared {cleared} working fact(s)). Expected: {self.task_state.expected_action}")

    def plan_task(self) -> str:
        """
        Draft a concrete plan for the freshly-started task during the PLANNING
        stage. The plan is stored as a planning-stage note (so it survives pauses
        and reloads) and the expected action becomes "review & approve, then
        /advance" — execution is not meant to start until the plan is agreed.
        """
        if not self.task_state.is_active():
            return "No active task to plan."
        messages = self._build_context()
        messages.append({
            "role": "user",
            "content": (
                f'We are at the PLANNING stage of the task: "{self.task_state.name}". '
                "Produce a concrete, numbered, step-by-step plan to accomplish it. "
                "Be specific and concise. Do NOT start doing the work yet — this is the plan only."
            ),
        })
        try:
            response = self._client.chat.completions.create(model=self._model, messages=messages)
            plan = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"(could not generate a plan: {exc})"
        if plan:
            self.task_state.add_note(f"proposed plan:\n{plan}")
            self.task_state.step = "review & approve the proposed plan"
            self.task_state.expected_action = (
                "Review the proposed plan; once you approve it, /advance to execution."
            )
            self._save_task_state()
        return plan

    def run_stage(self) -> str:
        """
        Do the work of the CURRENT stage via the model. Called right after entering
        'execution' (carry out the plan) or 'validation' (check the result), so the
        stage transition actually produces work instead of sitting idle. The output
        is stored as a stage note (persisted → survives pauses/reloads), and the
        expected action points at the next /advance.
        Returns "" for stages with no auto-work (planning, done).
        """
        if not self.task_state.is_active():
            return "No active task."
        stage = self.task_state.stage
        prompt = STAGE_WORK_PROMPT.get(stage)
        if not prompt:
            return ""
        messages = self._build_context()
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(model=self._model, messages=messages)
            out = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"(could not run the {stage} stage: {exc})"
        if out:
            self.task_state.add_note(f"{stage} output:\n{out}")
            next_hint = {
                "execution": "Review the execution output; /advance to validation.",
                "validation": "Review the validation result; /advance to mark the task done.",
            }
            self.task_state.expected_action = next_hint.get(stage, self.task_state.expected_action)
            self.task_state.step = f"review the {stage} output"
            self._save_task_state()
        return out

    def advance_stage(self) -> str:
        _, msg = self.task_state.advance()
        self._save_task_state()
        return msg

    def back_stage(self) -> str:
        _, msg = self.task_state.back()
        self._save_task_state()
        return msg

    def pause_task(self) -> str:
        msg = self.task_state.pause()
        self._save_task_state()
        return msg

    def resume_task(self) -> str:
        msg = self.task_state.resume()
        self._save_task_state()
        return msg

    def set_step(self, text: str) -> str:
        msg = self.task_state.set_step(text)
        self._save_task_state()
        return msg

    def set_expected(self, text: str) -> str:
        msg = self.task_state.set_expected(text)
        self._save_task_state()
        return msg

    def add_task_note(self, text: str) -> str:
        msg = self.task_state.add_note(text)
        self._save_task_state()
        return msg

    def end_task(self) -> str:
        old = self.task or self.task_state.name
        cleared = self.working.clear()
        self.task_state = TaskState()
        self._task_state_path.unlink(missing_ok=True)
        return (f"Ended task '{old or '(none)'}', cleared {cleared} working item(s) "
                f"and reset the state machine.")

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
        # The task identity now lives in the FSM block, so the working block only
        # carries the task DATA (the task name marker is excluded here).
        data = [f.content for f in self.working.all() if not f.content.startswith(TASK_PREFIX)]
        if not data:
            return ""
        return "Task data:\n" + "\n".join(f"  - {c}" for c in data)

    def _build_context(self) -> list[dict]:
        messages: list[dict] = [self._system]

        # Personalisation is connected to EVERY request, right after the system prompt.
        profile_block = self.profile.to_block()
        if profile_block:
            messages.append({"role": "system", "content": profile_block})

        # The task state machine is also connected to EVERY request — this is what
        # lets the assistant continue after a pause without re-explanation.
        state_block = self.task_state.to_block()
        if state_block:
            messages.append({"role": "system", "content": state_block})

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
        # HARD PAUSE: while the task is paused the agent refuses to work — it does
        # NOT route facts, does NOT call the model and does NOT touch the dialog.
        # The only way forward is /resume.
        if self.task_state.is_active() and self.task_state.paused:
            msg = (f"⏸ Task '{self.task_state.name}' is paused at stage "
                   f"'{self.task_state.stage}'. Type /resume to continue.")
            stats = TurnStats(
                request_tokens=0,
                response_tokens=0,
                history_tokens=_estimate_tokens([self._system] + self.short_term.all()),
                routed=[],
                layer_counts=self.layer_counts(),
                interrupted=False,
            )
            return msg, stats

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

        # 3. assemble context (system + profile + FSM + memory + dialog) and call the model
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
            "stage": self.task_state.stage if self.task_state.is_active() else "(none)",
            "step": self.task_state.step or "(none)",
            "expected_action": self.task_state.expected_action if self.task_state.is_active() else "(none)",
            "paused": self.task_state.paused,
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
    paused = " (PAUSED)" if s["paused"] else ""
    print(
        f"[task: {s['task']} | stage: {s['stage']}{paused} | "
        f"short-term: {s['short_term_messages']} msgs | "
        f"working: {s['working_facts']} facts | "
        f"long-term: {s['long_term_facts']} facts {s['long_term_by_category']}]"
    )
    print(f"[profile: {s['profile']}]\n")


def _print_state(agent: MemoryAgent) -> None:
    ts = agent.task_state
    if not ts.is_active():
        print("[state] (no active task — start one with /task <name>)\n")
        return
    paused = "  [PAUSED]" if ts.paused else ""
    print("[task state machine]")
    print(f"  Task:  {ts.name}")
    print(f"  Stage: {ts.stage_track()}{paused}")
    print(f"  Step:  {ts.step or '(not set)'}")
    print(f"  Next:  {ts.expected_action}")
    if ts.notes:
        print(f"  Progress: {len(ts.notes)} note(s) recorded")
    print()


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

    print(f"Model: {MODEL}  |  three-layer memory + profile + task state machine")
    print("Stages: planning → execution → validation → done")
    print("Commands: /memory /state /facts <layer> /remember <layer> <text>")
    print("          /task <name> /advance /back /step <text> /expect <text> /note <text>")
    print("          /pause /resume /endtask /forget <layer> /reset /profile ... /quit")
    print("Tip: press Ctrl-C while the agent is replying to interrupt and auto-pause.")
    _print_status(agent)
    if agent.task_state.is_active():
        # We loaded an in-flight task — show where we left off so the user can resume.
        _print_state(agent)

    try:
        _chat_loop(agent)
    finally:
        # Short-term memory is the current dialog only — wipe it on exit.
        # The task state machine is deliberately NOT cleared, so the task can be
        # resumed in a later session exactly where it was paused.
        agent.short_term.clear()
        print("(short-term dialog cleared; task state preserved for resume)")


def _chat_loop(agent: MemoryAgent) -> None:
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not user_input:
            continue

        # Robust exit: tolerate invisible chars (zero-width/BOM/NBSP that survive
        # str.strip()), case and trailing punctuation, so /quit always exits.
        _INVISIBLE = "\u200b\u200c\u200d\ufeff\xa0 \t\r\n"
        cleaned = user_input.strip(_INVISIBLE).lower().rstrip(" .!?")
        if cleaned in ("/quit", "/exit"):
            print("Bye!")
            return
        if user_input in ("/memory", "/status"):
            _print_status(agent)
            continue
        if user_input in ("/state", "/fsm"):
            _print_state(agent)
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
            print(agent.start_task(name))
            print("Drafting a plan…")
            plan = agent.plan_task()
            print(f"\n[proposed plan — review it, then /advance to start execution]\n{plan}\n")
            continue
        if user_input in ("/advance", "/next"):
            print(agent.advance_stage())
            # Entering execution/validation immediately does that stage's work.
            if agent.task_state.stage in STAGE_WORK_PROMPT:
                stage = agent.task_state.stage
                print(f"Working on the {stage} stage…")
                out = agent.run_stage()
                print(f"\n[{stage}]\n{out}\n")
            else:
                print()
            continue
        if user_input in ("/back", "/prev"):
            print(agent.back_stage())
            # Re-entering execution/validation re-does that stage's work.
            if agent.task_state.stage in STAGE_WORK_PROMPT:
                stage = agent.task_state.stage
                print(f"Working on the {stage} stage…")
                out = agent.run_stage()
                print(f"\n[{stage}]\n{out}\n")
            else:
                print()
            continue
        if user_input.startswith("/step"):
            text = user_input[len("/step"):].strip()
            print(agent.set_step(text) + "\n")
            continue
        if user_input.startswith("/expect"):
            text = user_input[len("/expect"):].strip()
            print(agent.set_expected(text) + "\n")
            continue
        if user_input.startswith("/note"):
            text = user_input[len("/note"):].strip()
            print(agent.add_task_note(text) + "\n")
            continue
        if user_input == "/pause":
            print(agent.pause_task() + "\n")
            continue
        if user_input == "/resume":
            print(agent.resume_task() + "\n")
            continue
        if user_input == "/endtask":
            print(agent.end_task() + "\n")
            continue
        if user_input.startswith("/forget"):
            layer = user_input[len("/forget"):].strip()
            print(agent.forget(layer) + "\n")
            continue

        # Stream the reply token-by-token so a long generation can be aborted with
        # Ctrl-C at ANY moment — on abort the task is automatically hard-paused.
        streamed = {"any": False}

        def _on_token(chunk: str) -> None:
            if not streamed["any"]:
                print("Agent: ", end="", flush=True)
                streamed["any"] = True
            print(chunk, end="", flush=True)

        reply, stats = agent.chat(user_input, on_token=_on_token)
        if streamed["any"]:
            print()  # end the streamed line
        else:
            print(f"Agent: {reply}")  # non-streamed reply (e.g. the paused stub)

        if stats.interrupted:
            print("\n[generation interrupted]")
            if agent.task_state.is_active():
                print(agent.pause_task())
        if stats.routed:
            saved = "; ".join(f"{r['layer']}: {r['content']}" for r in stats.routed)
            print(f"  ↳ saved → {saved}")
        print(
            f"[tokens — req: {stats.request_tokens} | res: {stats.response_tokens} | "
            f"raw-history: ~{stats.history_tokens} | layers: {stats.layer_counts}]\n"
        )


if __name__ == "__main__":
    main()
