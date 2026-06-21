"""
Tests for Task 13 — the task modelled as a finite state machine (FSM).

Part 1: Unit tests — the FSM itself (stages, step, expected action, pause/resume,
        progress notes, persistence), and its integration into the agent
        (start/advance/end, the state block injected into every request, and —
        the headline feature — RESUME ACROSS A RELOAD without re-explanation).
Part 2: Integration case (real DeepSeek) — pause at the 'execution' stage, then
        a brand-new agent instance (simulating a restart) resumes from the saved
        state and answers "what's next?" grounded in the plan, without the task
        being re-explained.

Run unit tests:        pytest test_task13.py -v
Run integration case:  pytest test_task13.py -m integration -v -s
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from task13 import (
    FactFile,
    ShortTermMemory,
    MemoryAgent,
    MemoryRouter,
    UserProfile,
    TaskState,
    STAGES,
    STAGE_EXPECTED_ACTION,
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


def _reload(tmp_path) -> MemoryAgent:
    """A fresh agent over the SAME memory dir — simulates closing & reopening."""
    return MemoryAgent(_make_mock_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")


def _chat_calls(agent: MemoryAgent):
    """Non-router create() calls."""
    calls = agent._client.chat.completions.create.call_args_list
    return [
        c for c in calls
        if "route facts" not in (c[1].get("messages", [{}])[0].get("content", "") or "").lower()
    ]


def _last_blob(agent: MemoryAgent) -> str:
    return " ".join(m["content"] for m in _chat_calls(agent)[-1][1]["messages"])


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — UNIT TESTS: the finite state machine
# ══════════════════════════════════════════════════════════════════════════════


class TestStagesAreExact:
    def test_stages_are_exactly_as_specified(self):
        # The task requires EXACTLY these states, in this order.
        assert STAGES == ("planning", "execution", "validation", "done")

    def test_every_stage_has_an_expected_action(self):
        for s in STAGES:
            assert STAGE_EXPECTED_ACTION[s]


class TestTaskStateMachine:
    def test_inactive_until_named(self):
        assert not TaskState().is_active()
        assert TaskState().to_block() == ""
        ts = TaskState(name="T")
        assert ts.is_active()

    def test_starts_at_planning_with_default_action(self):
        ts = TaskState(name="T")
        assert ts.stage == "planning"
        assert ts.expected_action == STAGE_EXPECTED_ACTION["planning"]
        assert ts.paused is False

    def test_advance_walks_the_pipeline_in_order(self):
        ts = TaskState(name="T")
        ok, _ = ts.advance(); assert ok and ts.stage == "execution"
        ok, _ = ts.advance(); assert ok and ts.stage == "validation"
        ok, _ = ts.advance(); assert ok and ts.stage == "done"
        # Cannot advance past the final stage.
        ok, msg = ts.advance()
        assert not ok and ts.stage == "done"
        assert "final stage" in msg

    def test_back_walks_the_pipeline_in_reverse(self):
        ts = TaskState(name="T")
        ts.advance(); ts.advance()        # → validation
        ok, _ = ts.back(); assert ok and ts.stage == "execution"
        ok, _ = ts.back(); assert ok and ts.stage == "planning"
        # Cannot go back past the first stage.
        ok, msg = ts.back()
        assert not ok and ts.stage == "planning"
        assert "first stage" in msg

    def test_back_resets_step_and_expected_action(self):
        ts = TaskState(name="T")
        ts.advance()                      # execution
        ts.set_step("create index")
        ts.back()                         # → planning
        assert ts.stage == "planning"
        assert ts.step == ""
        assert ts.expected_action == STAGE_EXPECTED_ACTION["planning"]
        assert any("create index" in n for n in ts.notes)  # work preserved as a note

    def test_back_on_inactive_is_rejected(self):
        ok, msg = TaskState().back()
        assert not ok and "No active task" in msg

    def test_advance_updates_expected_action_and_resets_step(self):
        ts = TaskState(name="T")
        ts.set_step("draft the plan")
        ts.advance()
        assert ts.expected_action == STAGE_EXPECTED_ACTION["execution"]
        assert ts.step == ""  # step is per-stage; cleared on transition

    def test_advance_preserves_step_as_a_progress_note(self):
        ts = TaskState(name="T")
        ts.set_step("agreed on index + matview approach")
        ts.advance()
        assert any("agreed on index" in n for n in ts.notes)
        assert ts.notes[0].startswith("planning:")

    def test_advance_on_inactive_is_rejected(self):
        ok, msg = TaskState().advance()
        assert not ok and "No active task" in msg

    def test_pause_and_resume_toggle_at_any_stage(self):
        ts = TaskState(name="T")
        ts.advance()  # now at execution
        assert "Paused" in ts.pause()
        assert ts.paused is True
        assert "already paused" in ts.pause()  # idempotent-ish
        assert "Resumed" in ts.resume()
        assert ts.paused is False
        assert "already running" in ts.resume()

    def test_notes_are_tagged_with_current_stage(self):
        ts = TaskState(name="T")
        ts.add_note("plan A chosen")
        ts.advance()
        ts.add_note("index created")
        assert ts.notes == ["planning: plan A chosen", "execution: index created"]

    def test_set_expected_falls_back_to_stage_default_when_empty(self):
        ts = TaskState(name="T")
        ts.set_expected("custom action")
        assert ts.expected_action == "custom action"
        ts.set_expected("")
        assert ts.expected_action == STAGE_EXPECTED_ACTION["planning"]

    def test_to_block_shows_stage_step_action_and_pause(self):
        ts = TaskState(name="Speed up dashboard")
        ts.set_step("add composite index")
        ts.add_note("plan agreed")
        ts.pause()
        block = ts.to_block()
        assert "Speed up dashboard" in block
        assert "[planning]" in block               # current stage highlighted in the track
        assert "add composite index" in block      # current step
        assert STAGE_EXPECTED_ACTION["planning"] in block  # expected action
        assert "PAUSED" in block
        assert "plan agreed" in block               # progress note carried along

    def test_save_load_roundtrip_preserves_everything(self, tmp_path):
        path = tmp_path / "task_state.json"
        ts = TaskState(name="T")
        ts.advance()                 # execution
        ts.set_step("create index")
        ts.add_note("plan ready")
        ts.pause()
        ts.save(path)

        loaded = TaskState.load(path)
        assert loaded.name == "T"
        assert loaded.stage == "execution"
        assert loaded.step == "create index"
        assert loaded.paused is True
        assert loaded.notes  # the planning note + any from advance

    def test_load_missing_is_inactive(self, tmp_path):
        assert not TaskState.load(tmp_path / "nope.json").is_active()

    def test_load_invalid_stage_defaults_to_planning(self, tmp_path):
        path = tmp_path / "ts.json"
        path.write_text('{"name": "T", "stage": "bogus"}', encoding="utf-8")
        assert TaskState.load(path).stage == "planning"


# ══════════════════════════════════════════════════════════════════════════════
# PART 1b — the FSM wired into the agent
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentTaskLifecycle:
    def test_start_task_initialises_fsm_at_planning(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("Speed up dashboard")
        assert agent.task == "Speed up dashboard"              # name in working layer
        assert agent.task_state.name == "Speed up dashboard"   # name in FSM
        assert agent.task_state.stage == "planning"
        # FSM persisted to its own JSON file
        assert (tmp_path / "memory" / "task_state.json").exists()
        # task name still lives as the first working line (task 12 behaviour kept)
        working = (tmp_path / "memory" / "working.txt").read_text()
        assert working.splitlines()[0] == f"{TASK_PREFIX}Speed up dashboard"

    def test_task_command_drafts_and_stores_a_plan(self, tmp_path):
        agent = _agent(tmp_path, chat_reply="1. profile the slow query\n2. add an index")
        agent.start_task("Speed up dashboard")
        plan = agent.plan_task()
        assert "add an index" in plan
        # stored in the planning stage, with an "approve then advance" expectation
        assert agent.task_state.stage == "planning"
        assert any("add an index" in n for n in agent.task_state.notes)
        assert "advance" in agent.task_state.expected_action.lower()
        # and it survives a reload (resume keeps the plan)
        assert any("add an index" in n for n in _reload(tmp_path).task_state.notes)

    def test_entering_execution_runs_the_plan(self, tmp_path):
        agent = _agent(tmp_path, chat_reply="ran step 1; created the index")
        agent.start_task("T")
        agent.advance_stage()                 # planning → execution
        out = agent.run_stage()
        assert "created the index" in out
        assert agent.task_state.stage == "execution"
        assert any("execution output" in n for n in agent.task_state.notes)
        assert "validation" in agent.task_state.expected_action.lower()

    def test_entering_validation_runs_validation(self, tmp_path):
        agent = _agent(tmp_path, chat_reply="all checks passed")
        agent.start_task("T")
        agent.advance_stage()                 # execution
        agent.advance_stage()                 # validation
        out = agent.run_stage()
        assert "all checks passed" in out
        assert any("validation output" in n for n in agent.task_state.notes)

    def test_run_stage_is_noop_on_planning_and_done(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T")
        assert agent.run_stage() == ""        # planning: plan is drafted at /task, not here
        agent.advance_stage(); agent.advance_stage(); agent.advance_stage()  # → done
        assert agent.task_state.stage == "done"
        assert agent.run_stage() == ""        # done: terminal, nothing to do

    def test_advance_through_agent_persists(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T")
        agent.advance_stage()
        assert agent.task_state.stage == "execution"
        assert _reload(tmp_path).task_state.stage == "execution"

    def test_back_through_agent_persists(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T")
        agent.advance_stage()             # execution
        agent.advance_stage()             # validation
        agent.back_stage()                # → execution
        assert agent.task_state.stage == "execution"
        assert _reload(tmp_path).task_state.stage == "execution"

    def test_end_task_resets_fsm_and_deletes_file(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T")
        agent.advance_stage()
        agent.end_task()
        assert not agent.task_state.is_active()
        assert not (tmp_path / "memory" / "task_state.json").exists()
        assert agent.task == ""

    def test_state_block_injected_into_every_request(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("Build ETL")
        agent.advance_stage()  # execution
        agent.set_step("create staging table")
        agent.chat("ok")
        agent.chat("continue")
        for call in _chat_calls(agent):
            blob = " ".join(m["content"] for m in call[1]["messages"])
            assert "task state machine" in blob.lower()
            assert "Build ETL" in blob
            assert "create staging table" in blob

    def test_no_task_means_no_state_block(self, tmp_path):
        agent = _agent(tmp_path)
        agent.chat("hello")
        assert "task state machine" not in _last_blob(agent).lower()

    def test_hard_pause_blocks_the_model(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T")
        agent.pause_task()
        before = agent.short_term.count()
        reply, stats = agent.chat("you there?")
        assert "paused" in reply.lower()
        assert _chat_calls(agent) == []                 # model never called while paused
        assert stats.request_tokens == 0
        assert agent.short_term.count() == before        # the turn is not recorded
        # …and after /resume the agent works again
        agent.resume_task()
        agent.chat("now?")
        assert _chat_calls(agent)                         # model called after resume

    def test_status_reports_stage_and_pause(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("T")
        agent.advance_stage()
        agent.pause_task()
        s = agent.status()
        assert s["stage"] == "execution"
        assert s["paused"] is True


class TestPauseThenResumeAcrossReload:
    """The headline feature: pause, 'close' the program, reopen → continue."""

    def test_full_state_survives_a_reload(self, tmp_path):
        a = _agent(tmp_path)
        a.start_task("Speed up the orders dashboard")
        a.add_task_note("approach: composite index + daily matview")
        a.advance_stage()                 # → execution
        a.set_step("create the composite index")
        a.pause_task()

        # Simulate closing and reopening the program.
        b = _reload(tmp_path)
        ts = b.task_state
        assert ts.name == "Speed up the orders dashboard"
        assert ts.stage == "execution"
        assert ts.step == "create the composite index"
        assert ts.paused is True
        assert any("composite index + daily matview" in n for n in ts.notes)

    def test_resumed_agent_carries_plan_into_context_without_reexplaining(self, tmp_path):
        a = _agent(tmp_path)
        a.start_task("Speed up the orders dashboard")
        a.add_task_note("approach: composite index + daily matview")
        a.advance_stage()
        a.set_step("create the composite index")
        a.pause_task()

        b = _reload(tmp_path)
        assert "Resumed" in b.resume_task()
        b.chat("what's the next step?")
        # The reloaded agent injects the saved plan/step into the request, so the
        # model can continue without the user re-explaining the task.
        blob = _last_blob(b)
        assert "composite index + daily matview" in blob   # the decided plan
        assert "create the composite index" in blob        # the in-flight step
        assert "execution" in blob                          # the resumed stage


class TestProfileAndMemoryStillWork:
    """Task 12 features must keep working alongside the new FSM."""

    def test_profile_still_injected(self, tmp_path):
        agent = _agent(tmp_path)
        agent.load_preset("analyst_ru")
        agent.chat("hi")
        assert "Russian" in _last_blob(agent)

    def test_full_context_has_profile_fsm_and_memory(self, tmp_path):
        agent = _agent(
            tmp_path,
            routed_json='{"facts": [{"layer": "profile", "content": "User name Dmitrii"}]}',
        )
        agent.load_preset("analyst_ru")
        agent.remember("knowledge", "Kafka 3.2 is the backbone")
        agent.start_task("Build ETL")
        agent.advance_stage()                          # execution
        agent.remember("working", "source table is raw_events")
        agent.chat("what's next?")
        blob = _last_blob(agent)
        assert "always honour these" in blob.lower()   # profile
        assert "task state machine" in blob.lower()    # FSM
        assert "execution" in blob                      # FSM stage
        assert "Dmitrii" in blob                        # long-term profile fact
        assert "Kafka 3.2" in blob                      # long-term knowledge
        assert "Build ETL" in blob                      # task name (FSM block)
        assert "raw_events" in blob                     # working data

    def test_router_still_routes(self, tmp_path):
        agent = _agent(
            tmp_path,
            routed_json='{"facts": [{"layer": "working", "content": "orders has 5M rows"}]}',
        )
        agent.start_task("T")
        agent.chat("orders has 5M rows")
        assert agent._working_data_count() == 1


def test_normalize():
    assert _normalize("  Hello   World ") == "hello world"


def test_presets_well_formed():
    for name, preset in PROFILE_PRESETS.items():
        p = UserProfile(**preset)
        assert not p.is_empty(), f"preset {name} is empty"
        assert p.to_block(), f"preset {name} renders no block"


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — INTEGRATION CASE (real DeepSeek): pause → resume → continue
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestPauseResumeContinuation:
    """
    Prove the headline behaviour end-to-end against the real model:

      1. Start a task, agree a plan (planning), advance to execution, set the
         current step, then PAUSE.
      2. Spin up a SEPARATE agent over the same memory (a restart) and RESUME.
      3. Ask "what's the very next thing?" WITHOUT re-stating the task.
         The answer must be grounded in the persisted plan/step — proving the
         work continues without any re-explanation.
    """

    @pytest.fixture(scope="class")
    def client(self):
        return _real_client()

    def test_resume_after_pause_needs_no_reexplanation(self, client, tmp_path_factory):
        mem = tmp_path_factory.mktemp("p13") / "memory"

        print("\n" + "=" * 72)
        print("FSM PAUSE/RESUME CASE — continue without re-explanation")
        print("=" * 72)

        # ── Session 1: plan, advance to execution, pause ──────────────────────────
        a = MemoryAgent(client, MODEL, SYSTEM_PROMPT, memory_dir=mem)
        a.start_task("Speed up the orders dashboard")
        a.remember("working", "the orders table has 820M rows and a full scan takes 9 minutes")
        a.add_task_note("agreed approach: add a composite index on (status, created_at) and a daily materialized view")
        print(f">> {a.advance_stage()}")            # planning → execution
        print(f">> {a.set_step('create the composite index on (status, created_at)')}")
        print(f">> {a.pause_task()}")
        assert a.task_state.paused and a.task_state.stage == "execution"

        # ── Session 2: a fresh agent over the same memory (a restart) ─────────────
        b = MemoryAgent(client, MODEL, SYSTEM_PROMPT, memory_dir=mem)
        assert b.task_state.stage == "execution", "state did not survive the reload"
        assert b.task_state.paused
        print(f">> (reloaded) {b.resume_task()}")

        # Ask for the next step WITHOUT re-explaining the task.
        question = "What's the very next thing I should do?"
        answer, _ = b.chat(question)
        print(f"\nQ: {question}\nA: {answer}\n")

        low = answer.lower()
        # The answer must be grounded in the persisted plan/step, not asking us
        # to re-explain what the task even is.
        assert "index" in low, "resumed agent lost the persisted plan/step"
        assert b.task_state.stage == "execution"  # resuming did not lose the stage

        print("=" * 72)
        print("PROVEN: task paused at 'execution', reloaded in a new agent, resumed,")
        print("and continued from the saved plan/step with no re-explanation.")
        print("=" * 72)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        agent = MemoryAgent(
            _make_mock_client(chat_reply="ok"),
            MODEL, SYSTEM_PROMPT, memory_dir=Path(d) / "memory",
        )
        print(agent.start_task("demo task"))
        print(agent.advance_stage())
        print(agent.set_step("do the thing"))
        print(agent.add_task_note("decided plan A"))
        print(agent.pause_task())
        print("state block:\n" + agent.task_state.to_block())
        print("status:", agent.status())
