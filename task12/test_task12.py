"""
Tests for Task 12 — personalisation (user profile) on top of three-layer memory.

Part 1: Unit tests — profile persistence, profile injected into every request,
        the task living in the WORKING layer (no task.txt), memory routing.
Part 2: Integration case (real DeepSeek) — the SAME question asked under two
        DIFFERENT profiles yields differently styled answers, and a constraint
        (language) is honoured automatically.

Run unit tests:        pytest test_task12.py -v
Run integration case:  pytest test_task12.py -m integration -v -s
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from task12 import (
    FactFile,
    ShortTermMemory,
    MemoryAgent,
    MemoryRouter,
    UserProfile,
    PROFILE_PRESETS,
    TASK_PREFIX,
    _normalize,
    load_api_key,
    MODEL,
    SYSTEM_PROMPT,
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


def _chat_calls(agent: MemoryAgent):
    """Non-router create() calls."""
    calls = agent._client.chat.completions.create.call_args_list
    return [
        c for c in calls
        if "route facts" not in (c[1].get("messages", [{}])[0].get("content", "") or "").lower()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestUserProfile:
    def test_empty_profile_renders_nothing(self):
        assert UserProfile().is_empty()
        assert UserProfile().to_block() == ""

    def test_block_contains_all_set_fields(self):
        p = UserProfile(
            name="Dmitrii", role="data engineer",
            style="concise", format="bullet points", language="Russian",
            constraints=["no fluff", "include numbers"],
        )
        block = p.to_block()
        assert "Dmitrii" in block and "data engineer" in block
        assert "concise" in block
        assert "bullet points" in block
        assert "Russian" in block
        assert "no fluff" in block and "include numbers" in block

    def test_set_field_validates(self):
        p = UserProfile()
        assert "Set style" in p.set_field("style", "friendly")
        assert p.style == "friendly"
        assert "Unknown field" in p.set_field("bogus", "x")

    def test_constraints_add_and_del(self):
        p = UserProfile()
        p.add_constraint("be brief")
        p.add_constraint("be brief")  # dup ignored
        assert p.constraints == ["be brief"]
        p.add_constraint("cite sources")
        assert "Removed" in p.del_constraint("1")        # by index
        assert p.constraints == ["cite sources"]
        assert "Removed" in p.del_constraint("cite sources")  # by text
        assert p.constraints == []

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "profile.json"
        p = UserProfile(name="X", style="formal", constraints=["a", "b"])
        p.save(path)
        loaded = UserProfile.load(path)
        assert loaded.name == "X"
        assert loaded.style == "formal"
        assert loaded.constraints == ["a", "b"]

    def test_load_missing_returns_empty(self, tmp_path):
        assert UserProfile.load(tmp_path / "nope.json").is_empty()


class TestProfileOnAgent:
    def test_profile_persists_to_json_file(self, tmp_path):
        agent = _agent(tmp_path)
        agent.set_profile_field("style", "concise")
        assert (tmp_path / "memory" / "profile.json").exists()
        # reload from disk
        agent2 = MemoryAgent(_make_mock_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")
        assert agent2.profile.style == "concise"

    def test_preset_loads_and_persists(self, tmp_path):
        agent = _agent(tmp_path)
        agent.load_preset("analyst_ru")
        assert agent.profile.language == "Russian"
        agent2 = MemoryAgent(_make_mock_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")
        assert agent2.profile.language == "Russian"

    def test_unknown_preset(self, tmp_path):
        agent = _agent(tmp_path)
        assert "Unknown preset" in agent.load_preset("nope")

    def test_clear_profile(self, tmp_path):
        agent = _agent(tmp_path)
        agent.load_preset("casual_en")
        agent.clear_profile()
        assert agent.profile.is_empty()
        assert not (tmp_path / "memory" / "profile.json").exists()

    def test_profile_injected_into_every_request(self, tmp_path):
        agent = _agent(tmp_path)
        agent.load_preset("analyst_ru")
        agent.chat("hello")
        agent.chat("again")
        # the personalisation block must appear in BOTH chat requests
        for call in _chat_calls(agent):
            blob = " ".join(m["content"] for m in call[1]["messages"])
            assert "always honour these" in blob.lower()
            assert "Russian" in blob

    def test_no_profile_means_no_block(self, tmp_path):
        agent = _agent(tmp_path)
        agent.chat("hello")
        blob = " ".join(m["content"] for m in _chat_calls(agent)[-1][1]["messages"])
        assert "always honour these" not in blob.lower()

    def test_different_profiles_produce_different_context(self, tmp_path):
        """Same code path, two profiles → two different personalisation blocks."""
        a1 = _agent(tmp_path / "a")
        a1.load_preset("analyst_ru")
        a1.chat("q")
        blob1 = " ".join(m["content"] for m in _chat_calls(a1)[-1][1]["messages"])

        a2 = _agent(tmp_path / "b")
        a2.load_preset("exec")
        a2.chat("q")
        blob2 = " ".join(m["content"] for m in _chat_calls(a2)[-1][1]["messages"])

        assert "Russian" in blob1 and "Russian" not in blob2
        assert "executive summary" in blob2


class TestTaskInWorkingLayer:
    def test_start_task_stores_in_working_no_task_file(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("Optimise orders")
        assert agent.task == "Optimise orders"
        # NO separate task.txt is ever created
        assert not (tmp_path / "memory" / "task.txt").exists()
        # the task lives as the first line of working.txt
        working = (tmp_path / "memory" / "working.txt").read_text()
        assert working.splitlines()[0] == f"{TASK_PREFIX}Optimise orders"

    def test_task_persists_via_working_across_reload(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("Migrate DWH")
        agent2 = MemoryAgent(_make_mock_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")
        assert agent2.task == "Migrate DWH"

    def test_start_task_clears_previous_working_data(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T1")
        agent.remember("working", "table has 5M rows")
        assert agent._working_data_count() == 1
        agent.start_task("T2")
        assert agent.task == "T2"
        assert agent._working_data_count() == 0  # old data gone

    def test_end_task_clears_working(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T")
        agent.remember("working", "data")
        msg = agent.end_task()
        assert "T" in msg
        assert agent.task == ""
        assert agent.working.count() == 0

    def test_task_appears_in_context(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("Build ETL")
        agent.remember("working", "source table is raw_events")
        agent.chat("what's next?")
        blob = " ".join(m["content"] for m in _chat_calls(agent)[-1][1]["messages"])
        assert "Build ETL" in blob
        assert "raw_events" in blob

    def test_working_count_excludes_task_marker(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T")
        agent.remember("working", "fact A")
        # status should count only real data, not the task marker line
        assert agent.status()["working_facts"] == 1
        assert agent.facts("working") == ["fact A"]


class TestMemoryRoutingStillWorks:
    def test_router_fact_lands_in_long_term(self, tmp_path):
        agent = _agent(
            tmp_path, chat_reply="hi",
            routed_json='{"facts": [{"layer": "profile", "content": "User is a data engineer"}]}',
        )
        _, stats = agent.chat("I'm a data engineer")
        assert agent.long_term.count() == 1
        assert agent._working_data_count() == 0
        assert stats.routed == [{"layer": "profile", "content": "User is a data engineer"}]

    def test_router_fact_lands_in_working(self, tmp_path):
        agent = _agent(
            tmp_path, chat_reply="ok",
            routed_json='{"facts": [{"layer": "working", "content": "table orders has 5M rows"}]}',
        )
        agent.chat("orders has 5M rows")
        assert agent._working_data_count() == 1
        assert agent.long_term.count() == 0

    def test_chit_chat_saves_nothing_durable(self, tmp_path):
        agent = _agent(tmp_path, chat_reply="hello", routed_json='{"facts": []}')
        agent.chat("hi there")
        assert agent.working.count() == 0
        assert agent.long_term.count() == 0
        assert agent.short_term.count() == 2

    def test_full_context_has_profile_memory_and_task(self, tmp_path):
        agent = _agent(
            tmp_path,
            routed_json='{"facts": [{"layer": "profile", "content": "User name Dmitrii"}]}',
        )
        agent.load_preset("analyst_ru")
        agent.remember("knowledge", "Kafka 3.2 is the backbone")
        agent.start_task("Build ETL")
        agent.remember("working", "source table is raw_events")
        agent.chat("what's next?")
        blob = " ".join(m["content"] for m in _chat_calls(agent)[-1][1]["messages"])
        assert "always honour these" in blob.lower()   # profile
        assert "Dmitrii" in blob                        # long-term profile fact
        assert "Kafka 3.2" in blob                      # long-term knowledge
        assert "Build ETL" in blob                      # working task
        assert "raw_events" in blob                     # working data


def test_normalize():
    assert _normalize("  Hello   World ") == "hello world"


def test_presets_well_formed():
    for name, preset in PROFILE_PRESETS.items():
        p = UserProfile(**preset)
        assert not p.is_empty(), f"preset {name} is empty"
        assert p.to_block(), f"preset {name} renders no block"


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — INTEGRATION CASE (real DeepSeek)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestPersonalisationCase:
    """
    The SAME memory + the SAME question, answered under TWO DIFFERENT profiles.

      Profile 1 — analyst_ru : Russian, concise, markdown bullet points.
      Profile 2 — exec       : English, high-level business summary, no code.

    Proves:
      • the profile is applied automatically to every request,
      • different profiles change the style/format/language of the answer,
      • a constraint (response language) is honoured without being asked.
    """

    @pytest.fixture(scope="class")
    def client(self):
        return _real_client()

    def test_two_profiles_two_styles(self, client, tmp_path_factory):
        mem = tmp_path_factory.mktemp("p12") / "memory"
        agent = MemoryAgent(client, MODEL, SYSTEM_PROMPT, memory_dir=mem)

        print("\n" + "=" * 72)
        print("PERSONALISATION CASE — same question, two profiles")
        print("=" * 72)

        # Shared, profile-agnostic context.
        agent.remember("profile", "The user works at RetailFlow on the orders pipeline.")
        agent.start_task("Speed up the orders dashboard")
        agent.remember("working", "the orders table has 820M rows and a full scan takes 9 minutes")

        question = "How should we make the orders dashboard load faster?"

        # ── Profile 1: Russian analyst, bullet points ─────────────────────────────
        print(f"\n>> {agent.load_preset('analyst_ru')}")
        agent.short_term.clear()
        ans_ru, _ = agent.chat(question)
        print(f"\n[P1 analyst_ru]\nQ: {question}\nA: {ans_ru}\n")

        # ── Profile 2: English exec summary ──────────────────────────────────────
        print(f">> {agent.load_preset('exec')}")
        agent.short_term.clear()
        ans_exec, _ = agent.chat(question)
        print(f"\n[P2 exec]\nA: {ans_exec}\n")

        # The Russian profile should yield Cyrillic text; the exec one should not.
        has_cyrillic_ru = any("Ѐ" <= ch <= "ӿ" for ch in ans_ru)
        has_cyrillic_exec = any("Ѐ" <= ch <= "ӿ" for ch in ans_exec)
        assert has_cyrillic_ru, "analyst_ru profile did not answer in Russian"
        assert not has_cyrillic_exec, "exec profile leaked Russian text"

        # Both answers should still be grounded in the shared working memory.
        assert "820" in ans_ru or "820" in ans_exec

        # The two answers must actually differ.
        assert ans_ru.strip() != ans_exec.strip()

        print("=" * 72)
        print("PROVEN: one shared memory, two profiles → two differently styled,")
        print("differently-languaged answers; the profile is applied automatically.")
        print("=" * 72)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        agent = MemoryAgent(
            _make_mock_client(
                chat_reply="ok",
                routed_json='{"facts": [{"layer": "profile", "content": "User is Dmitrii"}]}',
            ),
            MODEL, SYSTEM_PROMPT, memory_dir=Path(d) / "memory",
        )
        agent.load_preset("analyst_ru")
        agent.start_task("demo task")
        reply, stats = agent.chat("I'm Dmitrii")
        print("reply:", reply)
        print("routed:", stats.routed)
        print("task:", agent.task)
        print("status:", agent.status())
