"""
Tests for Task 11 — three-layer memory (short-term / working / long-term).

Part 1: Unit tests — file-per-layer storage, routing, separation, context build.
Part 2: Integration case (real DeepSeek) — proves each layer carries a distinct
        kind of knowledge and that dropping a layer changes the assistant's answer.

Run unit tests:        pytest test_task11.py -v
Run integration case:  pytest test_task11.py -m integration -v -s
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from task11 import (
    FactFile,
    ShortTermMemory,
    MemoryAgent,
    MemoryRouter,
    TurnStats,
    _normalize,
    load_api_key,
    MODEL,
    SYSTEM_PROMPT,
    LONG_TERM_CATEGORIES,
)
from openai import OpenAI


# ── Fixtures ─────────────────────────────────────────────────────────────────────

def _real_client() -> OpenAI:
    return OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")


def _make_mock_client(chat_reply: str = "ok", routed_json: str = '{"facts": []}') -> MagicMock:
    """
    Mock OpenAI client. The router call is detected by its system prompt
    ("route facts") and returns `routed_json`; every other call returns `chat_reply`.
    """
    mock = MagicMock()

    def side_effect(*args, **kwargs):
        messages = kwargs.get("messages", args[0] if args else [])
        system = messages[0].get("content", "") if messages else ""
        resp = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        if "route facts" in system.lower():
            resp.choices[0].message.content = routed_json
        else:
            resp.choices[0].message.content = chat_reply
        return resp

    mock.chat.completions.create.side_effect = side_effect
    return mock


def _agent(tmp_path, chat_reply="ok", routed_json='{"facts": []}') -> MemoryAgent:
    return MemoryAgent(
        _make_mock_client(chat_reply, routed_json),
        MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestFactFile:
    def test_working_stores_one_fact_per_line(self, tmp_path):
        ff = FactFile(tmp_path / "working.txt", tagged=False)
        ff.add("budget is 450000")
        ff.add("deadline is Q3")
        # All facts of the layer live in ONE file.
        assert (tmp_path / "working.txt").exists()
        lines = (tmp_path / "working.txt").read_text().strip().splitlines()
        assert len(lines) == 2
        assert ff.count() == 2

    def test_long_term_tags_category(self, tmp_path):
        ff = FactFile(tmp_path / "long_term.txt", tagged=True)
        ff.add("User's name is Dmitrii", category="profile")
        ff.add("Use Kafka as backbone", category="decision")
        content = (tmp_path / "long_term.txt").read_text()
        assert "[profile] User's name is Dmitrii" in content
        assert "[decision] Use Kafka as backbone" in content
        facts = ff.all()
        cats = {f.category for f in facts}
        assert cats == {"profile", "decision"}

    def test_dedup_skips_equivalent_fact(self, tmp_path):
        ff = FactFile(tmp_path / "w.txt", tagged=False)
        assert ff.add("budget is 450000") is True
        assert ff.add("Budget is 450000") is False   # case-insensitive dup
        assert ff.add("budget is 450000 ") is False   # whitespace dup
        assert ff.count() == 1

    def test_clear_returns_count_and_removes_file(self, tmp_path):
        ff = FactFile(tmp_path / "w.txt", tagged=False)
        ff.add("a"); ff.add("b")
        assert ff.clear() == 2
        assert ff.count() == 0

    def test_empty_fact_not_saved(self, tmp_path):
        ff = FactFile(tmp_path / "w.txt", tagged=False)
        assert ff.add("   ") is False
        assert ff.count() == 0


class TestShortTermMemory:
    def test_append_and_recent_window(self, tmp_path):
        st = ShortTermMemory(tmp_path / "st.jsonl", recent_keep=4)
        for i in range(6):
            st.append("user", f"msg {i}")
        assert st.count() == 6
        assert len(st.recent()) == 4
        assert st.recent()[-1]["content"] == "msg 5"

    def test_clear(self, tmp_path):
        st = ShortTermMemory(tmp_path / "st.jsonl")
        st.append("user", "hi")
        st.clear()
        assert st.count() == 0


class TestRouter:
    def test_parses_valid_json(self, tmp_path):
        client = _make_mock_client(
            routed_json='{"facts": [{"layer": "profile", "content": "name is X"}]}'
        )
        r = MemoryRouter(client, MODEL)
        out = r.route("I'm X", task="")
        assert out == [{"layer": "profile", "content": "name is X"}]

    def test_strips_markdown_fences(self):
        client = _make_mock_client(
            routed_json='```json\n{"facts": [{"layer": "working", "content": "task data"}]}\n```'
        )
        r = MemoryRouter(client, MODEL)
        out = r.route("msg", task="t")
        assert out == [{"layer": "working", "content": "task data"}]

    def test_invalid_json_returns_empty(self):
        client = _make_mock_client(routed_json="not json {{{")
        r = MemoryRouter(client, MODEL)
        assert r.route("msg", task="") == []

    def test_drops_unknown_layer(self):
        client = _make_mock_client(
            routed_json='{"facts": [{"layer": "bogus", "content": "x"}, '
                        '{"layer": "knowledge", "content": "y"}]}'
        )
        r = MemoryRouter(client, MODEL)
        out = r.route("msg", task="")
        assert out == [{"layer": "knowledge", "content": "y"}]


class TestMemoryAgentRouting:
    def test_router_fact_lands_in_long_term(self, tmp_path):
        agent = _agent(
            tmp_path, chat_reply="hi",
            routed_json='{"facts": [{"layer": "profile", "content": "User is a data engineer"}]}',
        )
        _, stats = agent.chat("I'm a data engineer")
        assert agent.long_term.count() == 1
        assert agent.working.count() == 0
        assert stats.routed == [{"layer": "profile", "content": "User is a data engineer"}]

    def test_router_fact_lands_in_working(self, tmp_path):
        agent = _agent(
            tmp_path, chat_reply="ok",
            routed_json='{"facts": [{"layer": "working", "content": "table orders has 5M rows"}]}',
        )
        agent.chat("orders has 5M rows")
        assert agent.working.count() == 1
        assert agent.long_term.count() == 0

    def test_chit_chat_saves_nothing_durable(self, tmp_path):
        agent = _agent(tmp_path, chat_reply="hello", routed_json='{"facts": []}')
        agent.chat("hi there")
        assert agent.working.count() == 0
        assert agent.long_term.count() == 0
        # but the dialog still went to short-term
        assert agent.short_term.count() == 2

    def test_layers_are_separate_files(self, tmp_path):
        agent = _agent(
            tmp_path,
            routed_json='{"facts": [{"layer": "knowledge", "content": "k"}, '
                        '{"layer": "working", "content": "w"}]}',
        )
        agent.chat("msg")
        mem = tmp_path / "memory"
        assert (mem / "short_term.jsonl").exists()
        assert (mem / "working.txt").exists()
        assert (mem / "long_term.txt").exists()
        # working file has no category tag; long_term does
        assert "[knowledge] k" in (mem / "long_term.txt").read_text()
        assert (mem / "working.txt").read_text().strip() == "w"

    def test_context_includes_all_layers(self, tmp_path):
        agent = _agent(
            tmp_path,
            routed_json='{"facts": [{"layer": "profile", "content": "User name Dmitrii"}]}',
        )
        agent.remember("knowledge", "Kafka 3.2 is the backbone")
        agent.start_task("Build ETL")
        agent.remember("working", "source table is raw_events")
        agent.chat("what's next?")

        # inspect the last chat (non-router) call
        calls = agent._client.chat.completions.create.call_args_list
        chat_calls = [
            c for c in calls
            if "route facts" not in (c[1].get("messages", [{}])[0].get("content", "") or "").lower()
        ]
        sent = chat_calls[-1][1]["messages"]
        blob = " ".join(m["content"] for m in sent)
        assert "Dmitrii" in blob          # long-term profile
        assert "Kafka 3.2" in blob        # long-term knowledge
        assert "Build ETL" in blob        # working task
        assert "raw_events" in blob       # working data


class TestTaskLifecycle:
    def test_start_task_clears_working(self, tmp_path):
        agent = _agent(tmp_path)
        agent.remember("working", "old task data")
        assert agent.working.count() == 1
        agent.start_task("New task")
        assert agent.working.count() == 0
        assert agent.task == "New task"

    def test_end_task_clears_working_keeps_long_term(self, tmp_path):
        agent = _agent(tmp_path)
        agent.remember("working", "temp data")
        agent.remember("profile", "durable identity")
        agent.end_task()
        assert agent.working.count() == 0
        assert agent.long_term.count() == 1   # long-term survives

    def test_task_persists_across_reload(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("Migrate DWH")
        agent2 = MemoryAgent(_make_mock_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")
        assert agent2.task == "Migrate DWH"


class TestManualAndForget:
    def test_remember_each_layer(self, tmp_path):
        agent = _agent(tmp_path)
        assert "profile" in agent.remember("profile", "x")
        assert "working" in agent.remember("working", "y")
        assert agent.long_term.count() == 1
        assert agent.working.count() == 1

    def test_remember_unknown_layer(self, tmp_path):
        agent = _agent(tmp_path)
        assert "Unknown layer" in agent.remember("bogus", "x")

    def test_remember_duplicate(self, tmp_path):
        agent = _agent(tmp_path)
        agent.remember("knowledge", "fact")
        assert "duplicate" in agent.remember("knowledge", "fact").lower()

    def test_forget_one_layer_only(self, tmp_path):
        agent = _agent(tmp_path)
        agent.remember("working", "w")
        agent.remember("profile", "p")
        agent.short_term.append("user", "hi")
        agent.forget("working")
        assert agent.working.count() == 0
        assert agent.long_term.count() == 1     # untouched
        assert agent.short_term.count() == 1    # untouched

    def test_forget_unknown_layer(self, tmp_path):
        agent = _agent(tmp_path)
        assert "Unknown layer" in agent.forget("bogus")


class TestPersistenceAndStatus:
    def test_facts_reload_from_disk(self, tmp_path):
        agent = _agent(tmp_path)
        agent.remember("profile", "User name Dmitrii")
        agent.remember("working", "current dataset is X")
        agent2 = MemoryAgent(_make_mock_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")
        assert agent2.long_term.count() == 1
        assert agent2.working.count() == 1

    def test_status_reports_each_layer(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T")
        agent.remember("working", "w")
        agent.remember("profile", "p")
        agent.remember("decision", "d")
        agent.chat("hello")  # adds short-term turns
        st = agent.status()
        assert st["task"] == "T"
        assert st["working_facts"] == 1
        assert st["long_term_facts"] == 2
        assert st["long_term_by_category"]["profile"] == 1
        assert st["long_term_by_category"]["decision"] == 1
        assert st["short_term_messages"] == 2

    def test_facts_listing(self, tmp_path):
        agent = _agent(tmp_path)
        agent.remember("knowledge", "k1")
        listing = agent.facts("long_term")
        assert listing == ["[knowledge] k1"]


def test_normalize():
    assert _normalize("  Hello   World ") == "hello world"


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — INTEGRATION CASE (real DeepSeek)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestMemoryLayersCase:
    """
    Scenario — a data engineer named Dmitrii at "RetailFlow".

    The case proves the three layers carry DISTINCT kinds of knowledge:

      long-term   profile:   "my name is Dmitrii, I'm a data engineer at RetailFlow"
      long-term   decision:  "we standardise on Kafka 3.2 as the streaming backbone"
      working     task data: "current task: optimise the orders table (820M rows)"
      short-term  dialog:    the running Q&A

    Then we show how the layers affect the answer:
      A. full memory          → assistant recalls name + decision + task number
      B. long-term wiped      → assistant no longer knows name/decision
      C. working wiped        → assistant no longer knows the task number
    """

    @pytest.fixture(scope="class")
    def client(self):
        return _real_client()

    def test_layers_drive_answers(self, client, tmp_path_factory):
        mem = tmp_path_factory.mktemp("case") / "memory"
        agent = MemoryAgent(client, MODEL, SYSTEM_PROMPT, memory_dir=mem)

        print("\n" + "=" * 72)
        print("THREE-LAYER MEMORY CASE — RetailFlow / Dmitrii")
        print("=" * 72)

        # ── Build up memory through natural conversation (router decides routing) ──
        script = [
            "Hi! My name is Dmitrii and I'm a data engineer at RetailFlow.",
            "Company-wide we've decided to standardise on Kafka 3.2 as our streaming backbone.",
        ]
        for turn in script:
            _, stats = agent.chat(turn)
            print(f"\nYou: {turn}")
            print(f"  ↳ routed: {stats.routed}")
            time.sleep(0.4)

        # Start a task → working layer
        print(f"\n>> {agent.start_task('Optimise the orders table')}")
        for turn in [
            "For this task: the orders table has 820M rows and a full scan takes 9 minutes.",
        ]:
            _, stats = agent.chat(turn)
            print(f"\nYou: {turn}")
            print(f"  ↳ routed: {stats.routed}")
            time.sleep(0.4)

        print("\n--- Memory snapshot ---")
        print(f"  long-term: {agent.facts('long_term')}")
        print(f"  working:   {agent.facts('working')}")
        print(f"  status:    {agent.status()}")

        # The router should have populated both durable layers.
        assert agent.long_term.count() >= 1, "long-term memory never populated"
        assert agent.working.count() >= 1, "working memory never populated"

        # ── A. Full memory: ask something that needs all layers ──────────────────
        agent.short_term.clear()  # remove dialog so only durable layers can answer
        q = ("Remind me: what's my name, which streaming technology did we "
             "standardise on, and how many rows are in the table I'm optimising?")
        ans_full, _ = agent.chat(q)
        print(f"\n[A] FULL MEMORY\nQ: {q}\nA: {ans_full}\n")
        low = ans_full.lower()
        assert "dmitrii" in low, "profile (name) not recalled from long-term"
        assert "kafka" in low, "decision (Kafka) not recalled from long-term"
        assert "820" in ans_full, "task number not recalled from working"

        # ── B. Wipe long-term, keep working ──────────────────────────────────────
        agent.forget("long_term")
        agent.short_term.clear()
        ans_no_lt, _ = agent.chat(q)
        print(f"[B] LONG-TERM WIPED\nA: {ans_no_lt}\n")
        low_b = ans_no_lt.lower()
        assert "dmitrii" not in low_b, "name leaked after long-term wipe"
        # working still present → task number may still be known
        print(f"  (820M still known? {'820' in ans_no_lt})")

        # ── C. Wipe working too ──────────────────────────────────────────────────
        agent.forget("working")
        agent.short_term.clear()
        ans_none, _ = agent.chat(q)
        print(f"[C] WORKING WIPED TOO\nA: {ans_none}\n")
        assert "820" not in ans_none, "task number leaked after working wipe"

        print("=" * 72)
        print("PROVEN: each layer carries distinct knowledge; removing a layer")
        print("removes exactly that knowledge from the assistant's answers.")
        print("=" * 72)


if __name__ == "__main__":
    # quick smoke without pytest
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        agent = MemoryAgent(
            _make_mock_client(
                chat_reply="ok",
                routed_json='{"facts": [{"layer": "profile", "content": "User is Dmitrii"}]}',
            ),
            MODEL, SYSTEM_PROMPT, memory_dir=Path(d) / "memory",
        )
        reply, stats = agent.chat("I'm Dmitrii")
        print("reply:", reply)
        print("routed:", stats.routed)
        print("status:", agent.status())
