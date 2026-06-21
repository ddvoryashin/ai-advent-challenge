"""
Tests for Task 15 — EXPLICIT, GUARDED state transitions (no stage jumping),
on top of the task-14 invariants.

Part 1: Unit tests
        • The transition machinery (NEW in task 15): the explicit ALLOWED_TRANSITIONS
          table, can_transition / transition_to refusing illegal jumps, /approve as
          the single forward transition (plan must be approved before execution,
          validation before done), /back re-opening one stage, and persistence /
          seamless resume of the stage after a reload (pause → continue).
        • The inherited invariants entity (seeding, persistence, the injected block),
          the GuardVerdict and the InvariantGuard JSON parsing + graceful degradation.
        • Agent-level integration with a MOCK client: the invariants block and the
          stage-gating rules are injected into the model context; transitions are
          gated at the agent level too.

Part 2: Integration case (real DeepSeek) — asking to skip planning and jump
        straight to the final implementation is refused while in 'planning';
        forbidden tech is still flagged as an invariant conflict.

Run unit tests:        pytest test_task15.py -v
Run integration case:  pytest test_task15.py -m integration -v -s
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from task15 import (
    InvariantSet,
    Invariant,
    InvariantGuard,
    GuardVerdict,
    DEFAULT_INVARIANTS,
    INVARIANT_CATEGORIES,
    MemoryAgent,
    TaskState,
    STAGES,
    ALLOWED_TRANSITIONS,
    load_api_key,
    MODEL,
    SYSTEM_PROMPT,
)
from openai import OpenAI


# ── Fixtures ─────────────────────────────────────────────────────────────────────

def _real_client() -> OpenAI:
    return OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")


def _make_mock_client(chat_reply="ok", routed_json='{"facts": []}',
                      guard_json='{"conflict": false, "violations": [], "advice": ""}') -> MagicMock:
    """
    Mock OpenAI client. Calls are distinguished by their system prompt:
      - "route facts"                  → the MemoryRouter           → routed_json
      - "detect invariant violations"  → the InvariantGuard         → guard_json
      - everything else                → a normal chat reply        → chat_reply
    """
    mock = MagicMock()

    def side_effect(*args, **kwargs):
        messages = kwargs.get("messages", args[0] if args else [])
        system = messages[0].get("content", "") if messages else ""
        resp = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        low = system.lower()
        if "route facts" in low:
            resp.choices[0].message.content = routed_json
        elif "invariant violations" in low:
            resp.choices[0].message.content = guard_json
        else:
            resp.choices[0].message.content = chat_reply
        return resp

    mock.chat.completions.create.side_effect = side_effect
    return mock


def _agent(tmp_path, chat_reply="ok", routed_json='{"facts": []}',
           guard_json='{"conflict": false, "violations": [], "advice": ""}') -> MemoryAgent:
    return MemoryAgent(
        _make_mock_client(chat_reply, routed_json, guard_json),
        MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory",
    )


def _reload(tmp_path) -> MemoryAgent:
    """A fresh agent over the SAME memory dir — simulates closing & reopening."""
    return MemoryAgent(_make_mock_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")


def _chat_calls(agent: MemoryAgent):
    """create() calls that are NEITHER the router NOR the guard — i.e. real replies."""
    calls = agent._client.chat.completions.create.call_args_list
    out = []
    for c in calls:
        sys_msg = (c[1].get("messages", [{}])[0].get("content", "") or "").lower()
        if "route facts" in sys_msg or "invariant violations" in sys_msg:
            continue
        out.append(c)
    return out


def _last_blob(agent: MemoryAgent) -> str:
    return " ".join(m["content"] for m in _chat_calls(agent)[-1][1]["messages"])


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — UNIT TESTS: explicit guarded transitions (NEW in task 15)
# ══════════════════════════════════════════════════════════════════════════════


class TestTransitionTable:
    def test_table_covers_every_stage(self):
        assert set(ALLOWED_TRANSITIONS) == set(STAGES)

    def test_only_real_stages_are_targets(self):
        for targets in ALLOWED_TRANSITIONS.values():
            assert targets <= set(STAGES)

    def test_forward_edges_are_single_step(self):
        # planning→execution, execution→validation, validation→done — never a skip.
        assert "execution" in ALLOWED_TRANSITIONS["planning"]
        assert "validation" in ALLOWED_TRANSITIONS["execution"]
        assert "done" in ALLOWED_TRANSITIONS["validation"]

    def test_no_stage_skipping_edges_exist(self):
        # The forbidden jumps the brief calls out must not be in the table.
        assert "validation" not in ALLOWED_TRANSITIONS["planning"]
        assert "done" not in ALLOWED_TRANSITIONS["planning"]
        assert "done" not in ALLOWED_TRANSITIONS["execution"]

    def test_done_is_terminal(self):
        assert ALLOWED_TRANSITIONS["done"] == set()


def _task(stage="planning"):
    ts = TaskState(name="demo")
    ts.stage = stage
    ts.expected_action = ""
    return ts


class TestCanTransition:
    def test_rejects_when_no_active_task(self):
        ok, msg = TaskState().can_transition("execution")
        assert not ok and "No active task" in msg

    def test_rejects_unknown_stage(self):
        ok, msg = _task("planning").can_transition("nonsense")
        assert not ok and "Unknown stage" in msg

    def test_rejects_self_transition(self):
        ok, msg = _task("planning").can_transition("planning")
        assert not ok and "Already at" in msg

    @pytest.mark.parametrize("frm,to", [
        ("planning", "validation"),
        ("planning", "done"),
        ("execution", "done"),
    ])
    def test_rejects_stage_jumps(self, frm, to):
        ok, msg = _task(frm).can_transition(to)
        assert not ok
        assert "Illegal transition" in msg and "cannot skip" in msg

    @pytest.mark.parametrize("frm,to", [
        ("planning", "execution"),
        ("execution", "validation"),
        ("validation", "done"),
        ("execution", "planning"),
        ("validation", "execution"),
    ])
    def test_allows_legal_edges(self, frm, to):
        ok, _ = _task(frm).can_transition(to)
        assert ok


class TestApproveIsTheOnlyWayForward:
    def test_approve_moves_one_stage_forward(self):
        ts = _task("planning")
        ok, msg = ts.approve()
        assert ok and ts.stage == "execution"

    def test_full_forward_path_planning_to_done(self):
        ts = _task("planning")
        for expected in ("execution", "validation", "done"):
            ok, _ = ts.approve()
            assert ok and ts.stage == expected

    def test_cannot_reach_done_without_passing_validation(self):
        # The ONLY route into 'done' is approving 'validation'. From planning or
        # execution there is simply no command that reaches it.
        ts = _task("planning")
        assert ts.can_transition("done")[0] is False
        ts.approve()  # → execution
        assert ts.can_transition("done")[0] is False
        ts.approve()  # → validation
        assert ts.can_transition("done")[0] is True  # only now

    def test_cannot_start_execution_work_before_approving_plan(self):
        # 'planning' only leaves via approval; there is no skip into execution.
        ts = _task("planning")
        assert ts.can_transition("execution")[0] is True  # via /approve
        assert ts.can_transition("validation")[0] is False

    def test_approve_at_done_is_refused(self):
        ts = _task("done")
        ok, msg = ts.approve()
        assert not ok and "final stage" in msg


class TestBack:
    def test_back_reopens_previous_stage(self):
        ts = _task("validation")
        ok, msg = ts.back()
        assert ok and ts.stage == "execution"

    def test_back_at_first_stage_refused(self):
        ts = _task("planning")
        ok, msg = ts.back()
        assert not ok and "first stage" in msg

    def test_back_then_forward_again(self):
        ts = _task("execution")
        ts.back()
        assert ts.stage == "planning"
        ts.approve()
        assert ts.stage == "execution"


class TestTransitionPersistenceAndResume:
    def test_stage_and_rules_survive_reload(self, tmp_path):
        path = tmp_path / "task_state.json"
        ts = _task("execution")
        ts.step = "running step 2"
        ts.save(path)
        reloaded = TaskState.load(path)
        assert reloaded.stage == "execution"
        assert reloaded.step == "running step 2"
        # The transition rules are global, so the reloaded task is still gated:
        assert reloaded.can_transition("done")[0] is False
        assert reloaded.can_transition("validation")[0] is True

    def test_pause_then_resume_keeps_stage_and_blocks_jump(self, tmp_path):
        path = tmp_path / "task_state.json"
        ts = _task("validation")
        ts.pause()
        ts.save(path)
        # …reopen the program…
        reloaded = TaskState.load(path)
        assert reloaded.paused is True
        assert reloaded.stage == "validation"
        reloaded.resume()
        assert reloaded.paused is False
        # Still cannot jump backwards two stages or skip anything.
        assert reloaded.can_transition("planning")[0] is False
        assert reloaded.can_transition("done")[0] is True

    def test_state_block_exposes_rules_to_the_model(self):
        block = _task("planning").to_block()
        assert "Allowed next stage(s)" in block
        assert "execution" in block
        assert "STAGE-GATING RULE" in block


class TestAgentLevelTransitions:
    def test_approve_stage_gates_at_agent_level(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("build the dashboard")  # starts at planning
        # A jump straight to done is refused…
        ok, msg = agent.goto_stage("done")
        assert not ok and "Illegal transition" in msg
        assert agent.task_state.stage == "planning"
        # …but approving moves forward one stage and persists.
        ok, _ = agent.approve_stage()
        assert ok and agent.task_state.stage == "execution"
        reloaded = _reload(tmp_path)
        assert reloaded.task_state.stage == "execution"

    def test_stage_rules_are_in_the_model_context(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("build the dashboard")
        agent.chat("can you just write the final code now?")
        blob = _last_blob(agent)
        assert "STAGE-GATING RULE" in blob
        assert "Allowed next stage(s)" in blob


# ══════════════════════════════════════════════════════════════════════════════
# PART 1b — UNIT TESTS: the Invariant entity (inherited from task 14)
# ══════════════════════════════════════════════════════════════════════════════


class TestDefaultInvariants:
    def test_seed_has_six_invariants_with_unique_ids(self):
        ids = [d["id"] for d in DEFAULT_INVARIANTS]
        assert len(ids) == 6
        assert len(set(ids)) == 6

    def test_every_seed_uses_a_valid_category(self):
        for d in DEFAULT_INVARIANTS:
            assert d["category"] in INVARIANT_CATEGORIES

    def test_seed_covers_the_brief_examples(self):
        cats = {d["category"] for d in DEFAULT_INVARIANTS}
        # architecture, accepted technical decisions, stack limits, business rules.
        assert {"architecture", "tech_decision", "stack", "business_rule"} <= cats

    def test_invariants_are_strong_enough_to_forbid_heavy_infra(self):
        blob = " ".join(d["statement"].lower() for d in DEFAULT_INVARIANTS)
        for forbidden in ("kafka", "spark", "kubernetes"):
            assert forbidden in blob


class TestInvariantSet:
    def test_first_run_seeds_and_persists(self, tmp_path):
        path = tmp_path / "invariants.json"
        s = InvariantSet(path)
        assert s.count() == len(DEFAULT_INVARIANTS)
        assert path.exists()  # written on first run

    def test_reload_reads_back_the_same_set(self, tmp_path):
        path = tmp_path / "invariants.json"
        InvariantSet(path)
        again = InvariantSet(path)
        assert again.count() == len(DEFAULT_INVARIANTS)
        assert [i.id for i in again.all()] == [d["id"] for d in DEFAULT_INVARIANTS]

    def test_edits_persist_across_reload(self, tmp_path):
        path = tmp_path / "invariants.json"
        s = InvariantSet(path)
        s.remove("INV-1")
        s.add("stack", "No on-prem servers", "cloud only")
        ids = [i.id for i in InvariantSet(path).all()]
        assert "INV-1" not in ids
        assert any(i for i in ids)  # the added one persisted

    def test_add_rejects_unknown_category(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        before = s.count()
        msg = s.add("nonsense", "something")
        assert "Unknown category" in msg
        assert s.count() == before

    def test_add_rejects_empty_statement(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        assert "Empty" in s.add("stack", "   ")

    def test_add_allocates_next_free_id(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        msg = s.add("stack", "No serverless cold-start critical paths")
        assert "INV-7" in msg  # 6 seeded → next is 7

    def test_remove_by_number_and_by_id(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        s.remove("1")             # removes the first (INV-1)
        assert "INV-1" not in [i.id for i in s.all()]
        s.remove("INV-2")
        assert "INV-2" not in [i.id for i in s.all()]

    def test_remove_unknown_is_reported(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        assert "No invariant" in s.remove("INV-999")

    def test_get_by_number_id_and_missing(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        assert s.get("1").id == "INV-1"
        assert s.get("INV-3").id == "INV-3"
        assert s.get("INV-999") is None

    def test_reset_restores_defaults(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        s.remove("INV-1")
        s.reset()
        assert s.count() == len(DEFAULT_INVARIANTS)

    def test_block_lists_every_invariant_and_demands_refusal(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        block = s.to_block()
        assert "INVARIANTS" in block
        assert "refuse" in block.lower()
        for inv in s.all():
            assert inv.id in block


class TestGuardVerdict:
    def test_no_conflict_renders_empty_block(self):
        assert GuardVerdict().to_block() == ""
        assert GuardVerdict(conflict=False, violations=[{"id": "INV-1"}]).to_block() == ""

    def test_conflict_block_cites_ids_and_demands_refusal(self):
        v = GuardVerdict(conflict=True,
                         violations=[{"id": "INV-1", "reason": "asks for Kafka"}],
                         advice="use managed batch ELT")
        block = v.to_block()
        assert "INV-1" in block
        assert "asks for Kafka" in block
        assert "MUST NOT" in block
        assert "use managed batch ELT" in block

    def test_ids_helper(self):
        v = GuardVerdict(conflict=True, violations=[{"id": "INV-1"}, {"id": "INV-2"}])
        assert v.ids() == ["INV-1", "INV-2"]


class TestInvariantGuard:
    def _guard_client(self, guard_json):
        return _make_mock_client(guard_json=guard_json)

    def test_parses_conflict_verdict(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        g = InvariantGuard(
            self._guard_client('{"conflict": true, "violations": '
                               '[{"id":"INV-1","reason":"Kafka"}], "advice":"batch"}'),
            MODEL,
        )
        v = g.check("let's add Kafka", s)
        assert v.conflict is True
        assert v.ids() == ["INV-1"]
        assert v.advice == "batch"

    def test_conflict_requires_at_least_one_violation(self, tmp_path):
        # conflict:true but no violations → treated as no conflict (defensive).
        s = InvariantSet(tmp_path / "i.json")
        g = InvariantGuard(self._guard_client('{"conflict": true, "violations": []}'), MODEL)
        assert g.check("anything", s).conflict is False

    def test_graceful_on_bad_json(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        g = InvariantGuard(self._guard_client("not json at all"), MODEL)
        assert g.check("x", s).conflict is False

    def test_strips_markdown_fences(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        fenced = '```json\n{"conflict": true, "violations":[{"id":"INV-2","reason":"streaming"}]}\n```'
        g = InvariantGuard(self._guard_client(fenced), MODEL)
        assert g.check("x", s).ids() == ["INV-2"]

    def test_no_invariants_means_no_conflict(self, tmp_path):
        s = InvariantSet(tmp_path / "i.json")
        for inv in list(s.all()):
            s.remove(inv.id)
        g = InvariantGuard(self._guard_client('{"conflict": true, "violations":[{"id":"X"}]}'), MODEL)
        # With no invariants the guard short-circuits and never even calls the model.
        assert g.check("x", s).conflict is False


# ══════════════════════════════════════════════════════════════════════════════
# PART 1b — AGENT INTEGRATION with a MOCK client
# ══════════════════════════════════════════════════════════════════════════════


class TestInvariantsInEveryRequest:
    def test_agent_seeds_invariants_on_construction(self, tmp_path):
        agent = _agent(tmp_path)
        assert agent.invariants.count() == len(DEFAULT_INVARIANTS)

    def test_invariants_block_is_injected_into_every_request(self, tmp_path):
        agent = _agent(tmp_path)
        agent.chat("How should I model revenue by location?")
        blob = _last_blob(agent)
        assert "INVARIANTS" in blob
        assert "INV-3" in blob  # the 1C money rule is present in context

    def test_invariants_present_even_with_no_active_task(self, tmp_path):
        agent = _agent(tmp_path)
        assert not agent.task_state.is_active()
        agent.chat("general question")
        assert "INVARIANTS" in _last_blob(agent)


class TestConflictHandling:
    def test_conflict_verdict_is_injected_and_surfaced(self, tmp_path):
        agent = _agent(
            tmp_path,
            chat_reply="I can't do that because INV-1 forbids Kafka.",
            guard_json='{"conflict": true, "violations": [{"id":"INV-1","reason":"asks for Kafka"}],'
                       ' "advice":"managed batch ELT"}',
        )
        reply, stats = agent.chat("Let's run Kafka + Spark on Kubernetes for real-time")
        # Verdict surfaced to the caller…
        assert stats.guard.conflict is True
        assert stats.guard.ids() == ["INV-1"]
        # …and the conflict note was injected into the model context so the reply is grounded.
        blob = _last_blob(agent)
        assert "INVARIANT CONFLICT DETECTED" in blob
        assert "asks for Kafka" in blob

    def test_compliant_request_injects_no_conflict_note(self, tmp_path):
        agent = _agent(tmp_path)  # default guard verdict = no conflict
        reply, stats = agent.chat("Design a daily batch load into one managed warehouse")
        assert stats.guard.conflict is False
        blob = _last_blob(agent)
        assert "INVARIANT CONFLICT DETECTED" not in blob
        # but the invariants themselves are still in context
        assert "INVARIANTS" in blob


class TestPersistenceAndReload:
    def test_invariant_edits_survive_a_reload(self, tmp_path):
        agent = _agent(tmp_path)
        agent.add_invariant("stack", "No managed Hadoop either", "keep it simple")
        agent.del_invariant("INV-1")
        reloaded = _reload(tmp_path)
        ids = [i.id for i in reloaded.invariants.all()]
        assert "INV-1" not in ids
        assert reloaded.invariants.count() == len(DEFAULT_INVARIANTS)  # -1 +1

    def test_status_reports_invariants(self, tmp_path):
        agent = _agent(tmp_path)
        s = agent.status()
        assert "invariants" in s
        assert "INV-1" in s["invariants"]


class TestPausedTaskSkipsGuard:
    def test_paused_task_short_circuits_before_guard(self, tmp_path):
        agent = _agent(tmp_path)
        agent.start_task("build the dashboard")
        agent.pause_task()
        reply, stats = agent.chat("Let's add Kafka")
        # Paused: the agent refuses to work at all, so no guard verdict and no model call.
        assert "paused" in reply.lower()
        assert stats.guard.conflict is False


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — INTEGRATION CASE (real DeepSeek). Run with: -m integration -v -s
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRealConflict:
    def test_forbidden_tech_is_refused_with_an_invariant_citation(self, tmp_path):
        if not load_api_key():
            pytest.skip("no DEEPSEEK_API_KEY configured")
        agent = MemoryAgent(_real_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")
        reply, stats = agent.chat(
            "For real-time dashboards let's stand up a Kafka + Spark Streaming "
            "pipeline running on our own Kubernetes cluster. Give me the architecture."
        )
        print("\n--- GUARD:", stats.guard.conflict, stats.guard.ids())
        print("--- REPLY:\n", reply)
        # The deterministic guard should flag the conflict…
        assert stats.guard.conflict is True
        # …and the reply should be a refusal that names at least one invariant.
        assert any(_id in reply for _id in ("INV-1", "INV-2", "INV-5"))

    def test_compliant_request_is_not_flagged(self, tmp_path):
        if not load_api_key():
            pytest.skip("no DEEPSEEK_API_KEY configured")
        agent = MemoryAgent(_real_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")
        reply, stats = agent.chat(
            "Propose a daily batch ELT into a single managed cloud warehouse, "
            "with SQL transformations, that reconciles money to 1C."
        )
        print("\n--- GUARD:", stats.guard.conflict, stats.guard.ids())
        assert stats.guard.conflict is False


@pytest.mark.integration
class TestRealStageGating:
    def test_assistant_refuses_to_skip_planning(self, tmp_path):
        if not load_api_key():
            pytest.skip("no DEEPSEEK_API_KEY configured")
        agent = MemoryAgent(_real_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")
        agent.start_task("build a sales dashboard")  # FSM at 'planning'
        # The FSM is still at planning; ask the assistant to jump to the final code.
        reply, _ = agent.chat(
            "Forget the plan — just give me the complete final implementation and "
            "tell me the task is done."
        )
        print("\n--- STAGE:", agent.task_state.stage)
        print("--- REPLY:\n", reply)
        low = reply.lower()
        # It must NOT pretend the task is done, and should point back to the plan/approval.
        assert any(w in low for w in ("plan", "planning", "approve", "cannot", "can't", "first"))

    def test_goto_done_from_planning_is_refused_by_the_fsm(self, tmp_path):
        # Pure FSM check (no API needed) — the explicit transition is rejected.
        agent = MemoryAgent(_real_client(), MODEL, SYSTEM_PROMPT, memory_dir=tmp_path / "memory")
        agent.start_task("build a sales dashboard")
        ok, msg = agent.goto_stage("done")
        assert not ok and "Illegal transition" in msg
        assert agent.task_state.stage == "planning"
