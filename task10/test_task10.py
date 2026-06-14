"""
Tests for Task 10 — three context management strategies.

Part 1: Unit / functional tests for each strategy.
Part 2: Data-mesh infrastructure case — quality, stability, token cost,
        and UX across all three strategies.

Data-mesh case — company: RetailFlow
─────────────────────────────────────
A mid-size e-commerce retailer. Current state:
  • Monolithic DWH on AWS Redshift (centralized)
  • 8 data engineers in one team — the bottleneck
  • 5 business domains: Orders, Customers, Inventory, Marketing, Payments
  • Stack: Airflow 2.4, dbt 1.6, Kafka 3.2, PostgreSQL 14, Grafana, Superset
  • Pain: 3-week lead time for new data requests, 23 active pipelines rotting

Goal: migrate to data mesh (domain ownership, data products,
      self-serve platform, federated governance).

Key facts planted across the scripted turns (used for recall scoring):
  lead_time      3 weeks           (turn 2)
  team_size      8 engineers       (turn 4)
  kafka_version  Kafka 3.2         (turn 6)
  budget         $450,000          (turn 8)
  compliance     GDPR + PCI DSS    (turn 10)
  timeline       18 months         (turn 12)
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from task10 import (
    BaseStrategy,
    BranchingStrategy,
    ContextAgent,
    SlidingWindowStrategy,
    StickyFactsStrategy,
    TokenStats,
    _estimate_tokens,
    load_api_key,
    MODEL,
    SYSTEM_PROMPT,
)
from openai import OpenAI

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _real_client() -> OpenAI:
    return OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")


def _make_mock_client(reply: str = "ok") -> MagicMock:
    """Returns an OpenAI-shaped mock whose chat.completions.create returns `reply`."""
    mock = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = reply
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    mock.chat.completions.create.return_value = resp
    return mock


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════


# ── Strategy 1: Sliding Window ────────────────────────────────────────────────

class TestSlidingWindow:
    def _strategy(self, window_size=4, reply="ok"):
        return SlidingWindowStrategy(
            _make_mock_client(reply), MODEL, SYSTEM_PROMPT, window_size=window_size
        )

    def test_single_turn_returns_reply(self):
        s = self._strategy()
        reply, stats = s.chat("hello")
        assert reply == "ok"
        assert isinstance(stats, TokenStats)
        assert stats.strategy == "sliding_window"

    def test_window_drops_old_messages(self):
        s = self._strategy(window_size=4)
        for i in range(6):
            s.chat(f"message {i}")
        assert len(s._history) == 12  # 6 user + 6 assistant
        # only last 4 should be sent to the API
        last_call = s._client.chat.completions.create.call_args
        sent_messages = last_call[1]["messages"]  # keyword arg
        # system prompt is always included
        assert sent_messages[0]["role"] == "system"
        # number of non-system messages ≤ window_size
        assert len(sent_messages) - 1 <= 4

    def test_window_size_one_keeps_only_last_message(self):
        s = self._strategy(window_size=2)
        for i in range(5):
            s.chat(f"turn {i}")
        last_call = s._client.chat.completions.create.call_args
        sent_messages = last_call[1]["messages"]
        non_system = [m for m in sent_messages if m["role"] != "system"]
        assert len(non_system) <= 2

    def test_status_reports_dropped_count(self):
        s = self._strategy(window_size=4)
        for i in range(5):
            s.chat(f"message {i}")
        st = s.status()
        assert st["total_messages"] == 10
        assert st["dropped"] == max(0, 10 - 4)

    def test_reset_clears_history(self):
        s = self._strategy()
        s.chat("hello")
        s.reset()
        assert s._history == []
        assert s.status()["total_messages"] == 0

    def test_token_stats_populated(self):
        s = self._strategy()
        _, stats = s.chat("test")
        assert stats.request_tokens == 10
        assert stats.response_tokens == 5
        assert stats.history_tokens > 0

    def test_persists_to_file(self, tmp_path):
        storage = tmp_path / "sw.json"
        s = SlidingWindowStrategy(
            _make_mock_client(), MODEL, SYSTEM_PROMPT, window_size=6, storage=storage
        )
        s.chat("hello")
        assert storage.exists()
        # reload
        s2 = SlidingWindowStrategy(
            _make_mock_client(), MODEL, SYSTEM_PROMPT, window_size=6, storage=storage
        )
        assert s2.status()["total_messages"] == 2


# ── Strategy 2: Sticky Facts ──────────────────────────────────────────────────

class TestStickyFacts:
    def _strategy(self, window_size=4, facts_reply='{"key": "value"}', chat_reply="ok"):
        mock = MagicMock()

        def side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            # facts extraction calls have a system message about extracting facts
            system_content = messages[0].get("content", "") if messages else ""
            resp = MagicMock()
            resp.usage.prompt_tokens = 10
            resp.usage.completion_tokens = 5
            if "Extract facts" in system_content:
                resp.choices[0].message.content = facts_reply
            else:
                resp.choices[0].message.content = chat_reply
            return resp

        mock.chat.completions.create.side_effect = side_effect
        return StickyFactsStrategy(mock, MODEL, SYSTEM_PROMPT, window_size=window_size)

    def test_facts_extracted_on_chat(self):
        s = self._strategy(facts_reply='{"goal": "data mesh migration"}')
        s.chat("We need to migrate to data mesh")
        assert "goal" in s.get_facts()
        assert s.get_facts()["goal"] == "data mesh migration"

    def test_facts_injected_into_context(self):
        s = self._strategy(facts_reply='{"cloud": "GCP"}')
        s.chat("We run on GCP only")
        s.chat("What next?")
        # find the actual chat API call (not facts extraction)
        calls = s._client.chat.completions.create.call_args_list
        chat_calls = [
            c for c in calls
            if "Extract facts" not in (c[1].get("messages", [{}])[0].get("content", "") or "")
        ]
        assert len(chat_calls) >= 2
        last_chat_messages = chat_calls[-1][1]["messages"]
        # facts should appear as a user message after system
        facts_present = any(
            "cloud" in m.get("content", "") or "GCP" in m.get("content", "")
            for m in last_chat_messages
        )
        assert facts_present

    def test_facts_survive_window_overflow(self):
        facts_reply = '{"key_fact": "important_value"}'
        s = self._strategy(window_size=2, facts_reply=facts_reply)
        for i in range(6):
            s.chat(f"message {i}")
        # facts persist even after window overflow
        assert "key_fact" in s.get_facts()

    def test_invalid_json_does_not_crash(self):
        s = self._strategy(facts_reply="not valid json {{{")
        reply, stats = s.chat("hello")
        assert reply == "ok"
        # facts stay empty or unchanged
        assert isinstance(s.get_facts(), dict)

    def test_status_includes_facts(self):
        s = self._strategy(facts_reply='{"deadline": "Q3"}')
        s.chat("deadline is Q3")
        st = s.status()
        assert "facts" in st
        assert "facts_count" in st

    def test_reset_clears_facts_and_history(self):
        s = self._strategy(facts_reply='{"x": "y"}')
        s.chat("something")
        s.reset()
        assert s.get_facts() == {}
        assert s._history == []

    def test_persists_facts_to_file(self, tmp_path):
        storage = tmp_path / "sf.json"
        mock = MagicMock()

        def side_effect(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            system_content = messages[0].get("content", "") if messages else ""
            resp = MagicMock()
            resp.usage.prompt_tokens = 10
            resp.usage.completion_tokens = 5
            if "Extract facts" in system_content:
                resp.choices[0].message.content = '{"saved": "fact"}'
            else:
                resp.choices[0].message.content = "reply"
            return resp

        mock.chat.completions.create.side_effect = side_effect
        s = StickyFactsStrategy(mock, MODEL, SYSTEM_PROMPT, storage=storage)
        s.chat("save a fact")
        assert storage.exists()
        s2 = StickyFactsStrategy(mock, MODEL, SYSTEM_PROMPT, storage=storage)
        assert "saved" in s2.get_facts()


# ── Strategy 3: Branching ─────────────────────────────────────────────────────

class TestBranching:
    def _strategy(self, reply="ok"):
        return BranchingStrategy(_make_mock_client(reply), MODEL, SYSTEM_PROMPT, window_size=4)

    def test_starts_on_main_branch(self):
        s = self._strategy()
        assert s._current == "main"
        assert "main" in s._branches

    def test_create_branch_forks_from_current(self):
        s = self._strategy()
        s.chat("first message")
        result, _ = s.chat("/branch feature-x")
        assert "feature-x" in result
        assert "feature-x" in s._branches
        assert s._current == "feature-x"

    def test_branch_inherits_parent_messages(self):
        s = self._strategy()
        s.chat("initial context")
        s.chat("more context")
        s.chat("/branch child")
        child = s._branches["child"]
        # child starts with parent's messages copied
        assert len(child.messages) == 4  # 2 user + 2 assistant from parent
        assert child.parent == "main"

    def test_branches_evolve_independently(self):
        s = self._strategy()
        s.chat("shared setup")
        s.chat("/branch branch-a")
        s.chat("branch-a message")
        s.chat("/switch main")
        main_len_before = len(s._branches["main"].messages)
        s.chat("main continues")
        assert len(s._branches["main"].messages) == main_len_before + 2
        # branch-a stays the same
        assert len(s._branches["branch-a"].messages) == 4  # 2 parent + 2 own

    def test_switch_branch(self):
        s = self._strategy()
        s.chat("/branch b1")
        result, _ = s.chat("/switch main")
        assert "main" in result
        assert s._current == "main"

    def test_switch_nonexistent_branch(self):
        s = self._strategy()
        result, _ = s.chat("/switch nonexistent")
        assert "not found" in result.lower() or "nonexistent" in result

    def test_list_branches(self):
        s = self._strategy()
        s.chat("/branch b1")
        s.chat("/switch main")
        s.chat("/branch b2")
        result, _ = s.chat("/branches")
        assert "main" in result
        assert "b1" in result
        assert "b2" in result
        assert "*" in result  # current marker

    def test_duplicate_branch_name_rejected(self):
        s = self._strategy()
        s.chat("/branch myb")
        result, _ = s.chat("/branch myb")
        assert "already exists" in result.lower()

    def test_window_applied_per_branch(self):
        s = BranchingStrategy(_make_mock_client(), MODEL, SYSTEM_PROMPT, window_size=2)
        for i in range(5):
            s.chat(f"msg {i}")
        last_call = s._client.chat.completions.create.call_args
        sent = last_call[1]["messages"]
        non_system = [m for m in sent if m["role"] != "system"]
        assert len(non_system) <= 2

    def test_reset_clears_all_branches(self):
        s = self._strategy()
        s.chat("/branch b1")
        s.reset()
        assert list(s._branches.keys()) == ["main"]
        assert s._current == "main"
        assert s._branches["main"].messages == []

    def test_persists_to_file(self, tmp_path):
        storage = tmp_path / "br.json"
        s = BranchingStrategy(_make_mock_client(), MODEL, SYSTEM_PROMPT, storage=storage)
        s.chat("hello")
        s.chat("/branch test-branch")
        assert storage.exists()
        s2 = BranchingStrategy(_make_mock_client(), MODEL, SYSTEM_PROMPT, storage=storage)
        assert "test-branch" in s2._branches
        assert s2._current == "test-branch"

    def test_status_reports_all_branches(self):
        s = self._strategy()
        s.chat("/branch b1")
        st = s.status()
        assert "main" in st["all_branches"]
        assert "b1" in st["all_branches"]
        assert st["current_branch"] == "b1"


# ── Context agent switcher ────────────────────────────────────────────────────

class TestContextAgent:
    def _agent(self):
        mock = _make_mock_client("response")
        # make facts extraction also return valid JSON
        original = mock.chat.completions.create.return_value

        def side_effect(*args, **kwargs):
            messages = kwargs.get("messages", [])
            system_content = messages[0].get("content", "") if messages else ""
            r = MagicMock()
            r.usage.prompt_tokens = 10
            r.usage.completion_tokens = 5
            if "Extract facts" in system_content:
                r.choices[0].message.content = '{"test": "value"}'
            else:
                r.choices[0].message.content = "response"
            return r

        mock.chat.completions.create.side_effect = side_effect
        return ContextAgent(mock, MODEL, SYSTEM_PROMPT, data_dir=Path("/tmp/task10_test"))

    def test_default_strategy_is_sliding_window(self):
        agent = self._agent()
        assert agent._current_strategy == "sliding_window"

    def test_switch_strategy(self):
        agent = self._agent()
        result, _ = agent.chat("/strategy sticky_facts")
        assert "sticky_facts" in result
        assert agent._current_strategy == "sticky_facts"

    def test_switch_to_all_strategies(self):
        agent = self._agent()
        for name in ContextAgent.STRATEGIES:
            result, _ = agent.chat(f"/strategy {name}")
            assert agent._current_strategy == name

    def test_switch_unknown_strategy(self):
        agent = self._agent()
        result, _ = agent.chat("/strategy nonexistent")
        assert "unknown" in result.lower() or "available" in result.lower()

    def test_status_command(self):
        agent = self._agent()
        result, _ = agent.chat("/status")
        assert "sliding_window" in result

    def test_reset_command(self):
        agent = self._agent()
        agent.chat("hello")
        result, _ = agent.chat("/reset")
        assert "reset" in result.lower()

    def test_strategies_are_independent(self):
        agent = self._agent()
        agent.chat("message on sliding_window")
        agent.chat("/strategy sticky_facts")
        # sticky_facts should have no history from sliding_window
        sf = agent.get_strategy("sticky_facts")
        assert sf.status()["total_messages"] == 0

    def test_branching_commands_pass_through(self):
        agent = self._agent()
        agent.chat("/strategy branching")
        result, _ = agent.chat("/branch topic-x")
        assert "topic-x" in result


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — DATA MESH CASE
# ══════════════════════════════════════════════════════════════════════════════

# Company: RetailFlow — e-commerce, 5 domains, wants data mesh.
# 6 facts planted across 12 turns; recall question at turn 13.

KEY_FACTS = {
    "lead_time":    "3 weeks",
    "team_size":    "8 engineers",
    "kafka_version": "Kafka 3.2",
    "budget":       "$450,000",
    "compliance":   "PCI DSS",
    "timeline":     "18 months",
}

DATA_MESH_SCRIPT: list[str] = [
    # Turn 1 — company intro (no planted facts)
    "We're RetailFlow, an e-commerce company. We want to adopt data mesh architecture. "
    "Our current DWH is on AWS Redshift. We have 5 business domains: "
    "Orders, Customers, Inventory, Marketing, Payments.",

    # Turn 2 — plant: lead_time (3 weeks)
    "Our central data team is overwhelmed. Every new data request takes 3 weeks to deliver. "
    "What's the first organizational change we should make for data mesh?",

    # Turn 3 — filler
    "How do we define domain boundaries for data products?",

    # Turn 4 — plant: team_size (8 engineers)
    "By the way, our entire data team has 8 engineers right now. "
    "How should we redistribute them across the 5 domains?",

    # Turn 5 — filler
    "What does a data product look like technically in a data mesh?",

    # Turn 6 — plant: kafka_version (Kafka 3.2)
    "Our event streaming runs on Kafka 3.2. "
    "How should we use Kafka as the backbone for data product publishing in the mesh?",

    # Turn 7 — filler
    "What self-serve data platform capabilities do we need to build first?",

    # Turn 8 — plant: budget ($450,000)
    "Our total transformation budget is $450,000 for this initiative. "
    "How should we allocate it across platform, tooling, and training?",

    # Turn 9 — filler
    "What federated governance policies are most critical in the first year?",

    # Turn 10 — plant: compliance (GDPR + PCI DSS)
    "Our Payments domain has strict GDPR and PCI DSS requirements. "
    "How do we handle compliance in a data mesh model without centralizing everything again?",

    # Turn 11 — filler
    "How do we measure the success of the data mesh adoption?",

    # Turn 12 — plant: timeline (18 months)
    "Our board approved an 18-month roadmap for the full migration. "
    "What should the milestone breakdown look like?",

    # Turn 13 — recall question
    "Final recap: what is our current data request lead time, how many engineers do we have, "
    "what Kafka version are we on, what is our transformation budget, "
    "what compliance standards apply to our Payments domain, "
    "and what is the approved timeline for this migration?",
]

RECALL_TURN = DATA_MESH_SCRIPT[-1]
REQUIRED_KEYWORDS = list(KEY_FACTS.values())


def score_recall(reply: str) -> tuple[int, list[str]]:
    found = [kw for kw in REQUIRED_KEYWORDS if kw.lower() in reply.lower()]
    return len(found), found


def _run_strategy(strategy: BaseStrategy, script: list[str], delay: float = 0.3) -> dict:
    """Run full script through a strategy; return metrics from the last turn."""
    total_req_tokens = 0
    total_res_tokens = 0
    final_reply = ""
    final_stats = None

    for i, turn in enumerate(script):
        print(f"  [{strategy.name}] turn {i+1}/{len(script)}...", end="\r", flush=True)
        reply, stats = strategy.chat(turn)
        total_req_tokens += stats.request_tokens
        total_res_tokens += stats.response_tokens
        final_reply = reply
        final_stats = stats
        if delay:
            time.sleep(delay)

    print()
    score, found = score_recall(final_reply)
    return {
        "strategy": strategy.name,
        "final_reply": final_reply,
        "recall_score": score,
        "keywords_found": found,
        "last_req_tokens": final_stats.request_tokens if final_stats else 0,
        "last_res_tokens": final_stats.response_tokens if final_stats else 0,
        "total_req_tokens": total_req_tokens,
        "total_res_tokens": total_res_tokens,
        "history_tokens": final_stats.history_tokens if final_stats else 0,
        "extra": final_stats.extra if final_stats else {},
    }


def _print_results(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("DATA MESH CASE — RESULTS")
    print("=" * 70)
    headers = ["Strategy", "Recall", "Last Req Tok", "Total Req Tok", "Hist Tok"]
    row_fmt = "{:<20} {:>8} {:>14} {:>15} {:>10}"
    print(row_fmt.format(*headers))
    print("-" * 70)
    for r in results:
        print(row_fmt.format(
            r["strategy"],
            f"{r['recall_score']}/{len(REQUIRED_KEYWORDS)}",
            r["last_req_tokens"],
            r["total_req_tokens"],
            r["history_tokens"],
        ))
    print("=" * 70)
    for r in results:
        found = r["keywords_found"]
        missing = [kw for kw in REQUIRED_KEYWORDS if kw not in found]
        print(f"\n[{r['strategy']}]")
        print(f"  Found:   {found}")
        print(f"  Missing: {missing}")
        print(f"  Extra:   {r['extra']}")
        print(f"  Reply snippet: {r['final_reply'][:300]}...")


@pytest.mark.integration
class TestDataMeshCase:
    """
    Integration test using real DeepSeek API.
    Run with: pytest -m integration -v -s

    Scenario
    ────────
    Company RetailFlow is planning a data mesh migration.
    The conversation has three phases:

      Phase 1 — main topic (data mesh):
        4 turns, each planting one concrete fact:
          • lead time:  "3 weeks"
          • team size:  "8 engineers"
          • budget:     "$450,000"
          • compliance: "PCI DSS"

      Phase 2 — off-topic tangent (CI/CD):
        4 unrelated turns about CI pipelines, secrets management, etc.
        These come right after the main facts and would fill a small window.

      Phase 3 — return and recall:
        One question asking the agent to recap all four planted facts.

    All three strategies run the same script with window=6.
    Branching additionally uses /branch and /switch to isolate the tangent.

    Dimensions compared
    ───────────────────
    Quality     recall score /4 on the final answer
    Stability   which facts survived — and which were lost
    Token cost  total request tokens across all turns
    UX          extra commands the user had to type
    """

    @pytest.fixture(scope="class")
    def client(self):
        return _real_client()

    def test_three_way_tangent_comparison(self, client):
        PLANTED_FACTS = ["3 weeks", "8 engineers", "$450,000", "PCI DSS"]
        WINDOW = 6

        MAIN_SETUP = [
            (
                "We are RetailFlow, an e-commerce company migrating to data mesh. "
                "Our DWH is on AWS Redshift, 5 business domains: Orders, Customers, "
                "Inventory, Marketing, Payments."
            ),
            (
                "Our biggest bottleneck: every new data request takes 3 weeks "
                "end-to-end. What organisational change should we make first?"
            ),
            (
                "We have exactly 8 data engineers today — all centralised. "
                "How should we redistribute them across the 5 domains?"
            ),
            (
                "Board approved a total transformation budget of $450,000. "
                "Our Payments domain must comply with PCI DSS. "
                "How do we handle compliance without re-centralising?"
            ),
        ]

        TANGENT = [
            "Completely different topic: best CI/CD approach for a Python monorepo?",
            "GitHub Actions vs GitLab CI — key trade-offs?",
            "How do we manage secrets in CI pipelines without storing them in git?",
            "Good git branching strategy for a 12-person dev team?",
        ]

        RECALL = (
            "Back to data mesh. Recap: what is our current data request lead time, "
            "how many data engineers do we have, what is the approved budget, "
            "and what compliance standard applies to the Payments domain?"
        )

        def _score(reply: str) -> tuple[int, list[str]]:
            found = [kw for kw in PLANTED_FACTS if kw.lower() in reply.lower()]
            return len(found), found

        results = {}

        # ── Strategy 1: Sliding Window ────────────────────────────────────────
        # All turns go into a single flat history. With window=6, the 4 tangent
        # turns (8 msgs) push most of the main turns outside the window.
        print("\n[1/3] Sliding Window (window=6)...")
        sw = SlidingWindowStrategy(client, MODEL, SYSTEM_PROMPT, window_size=WINDOW)
        total_sw = 0
        for turn in MAIN_SETUP:
            _, s = sw.chat(turn); total_sw += s.request_tokens; time.sleep(0.35)
        for turn in TANGENT:
            _, s = sw.chat(turn); total_sw += s.request_tokens; time.sleep(0.35)
        time.sleep(0.35)
        sw_reply, sw_s = sw.chat(RECALL); total_sw += sw_s.request_tokens
        sw_score, sw_found = _score(sw_reply)
        results["sliding_window"] = {
            "score": sw_score, "found": sw_found,
            "total_tokens": total_sw, "last_tokens": sw_s.request_tokens,
            "extra_commands": 0,
            "note": f"dropped {sw_s.extra['dropped']} msgs outside window",
            "reply": sw_reply,
        }
        print(f"  recall {sw_score}/4 | total tokens {total_sw} | last req {sw_s.request_tokens}")

        # ── Strategy 2: Sticky Facts ──────────────────────────────────────────
        # After each user turn, a separate LLM call extracts key-value facts.
        # Facts survive regardless of window size — they live outside the message list.
        # Downside: tangent facts also get extracted and mixed in.
        print("[2/3] Sticky Facts (window=6, auto fact-extraction each turn)...")
        sf = StickyFactsStrategy(client, MODEL, SYSTEM_PROMPT, window_size=WINDOW)
        total_sf = 0
        for turn in MAIN_SETUP:
            _, s = sf.chat(turn); total_sf += s.request_tokens; time.sleep(0.4)
        for turn in TANGENT:
            _, s = sf.chat(turn); total_sf += s.request_tokens; time.sleep(0.4)
        time.sleep(0.4)
        sf_reply, sf_s = sf.chat(RECALL); total_sf += sf_s.request_tokens
        sf_score, sf_found = _score(sf_reply)
        sf_facts = sf.get_facts()
        sf_pollution = any(
            kw in str(sf_facts).lower()
            for kw in ["ci/cd", "github actions", "secrets", "monorepo"]
        )
        results["sticky_facts"] = {
            "score": sf_score, "found": sf_found,
            "total_tokens": total_sf, "last_tokens": sf_s.request_tokens,
            "extra_commands": 0,
            "note": (
                f"{sf_s.extra['facts_count']} facts | "
                f"{'tangent mixed in' if sf_pollution else 'facts clean'}"
            ),
            "reply": sf_reply,
        }
        print(f"  recall {sf_score}/4 | total tokens {total_sf} | last req {sf_s.request_tokens} "
              f"| facts {sf_s.extra['facts_count']} | pollution={sf_pollution}")

        # ── Strategy 3: Branching ─────────────────────────────────────────────
        # User explicitly forks before the tangent (/branch) and returns after (/switch).
        # Main branch never sees the tangent messages → window stays over original facts.
        # Cost: user must issue 2 extra commands.
        print("[3/3] Branching (window=6, /branch + /switch around tangent)...")
        br = BranchingStrategy(client, MODEL, SYSTEM_PROMPT, window_size=WINDOW)
        total_br = 0
        for turn in MAIN_SETUP:
            _, s = br.chat(turn); total_br += s.request_tokens; time.sleep(0.35)

        main_before = len(br.current_branch.messages)
        br.chat("/branch ci-cd-tangent")   # extra command 1

        for turn in TANGENT:
            _, s = br.chat(turn); total_br += s.request_tokens; time.sleep(0.35)

        br.chat("/switch main")            # extra command 2
        assert len(br.current_branch.messages) == main_before, "main branch grew during tangent"

        time.sleep(0.35)
        br_reply, br_s = br.chat(RECALL); total_br += br_s.request_tokens
        br_score, br_found = _score(br_reply)

        # SRE/CI keywords must not appear in main branch messages (isolation check)
        main_content = " ".join(m["content"] for m in br.current_branch.messages)
        leaked = [kw for kw in ["GitHub Actions", "GitLab CI", "monorepo", "secrets"]
                  if kw.lower() in main_content.lower()]

        results["branching"] = {
            "score": br_score, "found": br_found,
            "total_tokens": total_br, "last_tokens": br_s.request_tokens,
            "extra_commands": 2,
            "note": (
                f"main={br_s.extra['branch_messages']} msgs | "
                f"{'LEAKED:' + str(leaked) if leaked else 'isolated'}"
            ),
            "reply": br_reply,
        }
        print(f"  recall {br_score}/4 | total tokens {total_br} | last req {br_s.request_tokens} "
              f"| leaked={leaked}")

        # ── Print comparison table ────────────────────────────────────────────
        stability = {0: "very low", 1: "low", 2: "medium", 3: "high", 4: "perfect"}
        sw_r, sf_r, br_r = results["sliding_window"], results["sticky_facts"], results["branching"]

        print("\n" + "=" * 76)
        print("DATA MESH TANGENT TEST — THREE-WAY COMPARISON  (window=6)")
        print("Company: RetailFlow | 4 main turns + 4 tangent turns | 4 facts to recall")
        print("=" * 76)
        fmt = "{:<30} {:>14} {:>14} {:>14}"
        print(fmt.format("Dimension", "SlidingWindow", "StickyFacts", "Branching"))
        print("-" * 76)
        print(fmt.format(
            "Quality  (recall /4)",
            f"{sw_r['score']}/4",
            f"{sf_r['score']}/4",
            f"{br_r['score']}/4",
        ))
        print(fmt.format(
            "Stability",
            stability[sw_r['score']],
            stability[sf_r['score']],
            stability[br_r['score']],
        ))
        print(fmt.format(
            "Token cost  (total)",
            sw_r['total_tokens'],
            sf_r['total_tokens'],
            br_r['total_tokens'],
        ))
        print(fmt.format(
            "Token cost  (last req)",
            sw_r['last_tokens'],
            sf_r['last_tokens'],
            br_r['last_tokens'],
        ))
        print(fmt.format(
            "UX  (extra commands)",
            sw_r['extra_commands'],
            sf_r['extra_commands'],
            f"{br_r['extra_commands']} (/branch,/switch)",
        ))
        print(fmt.format(
            "Context pollution",
            "tangent in window",
            "facts mixed" if sf_pollution else "facts clean",
            "isolated",
        ))
        print("-" * 76)
        print(fmt.format("Notes", sw_r['note'][:14], sf_r['note'][:14], br_r['note'][:14]))
        print("=" * 76)

        print("\nFacts found / missing per strategy:")
        for name, r in results.items():
            missing = [kw for kw in PLANTED_FACTS if kw not in r["found"]]
            print(f"  {name:<20} ✓ {r['found']}   ✗ {missing}")

        print("\nAccumulated Sticky Facts (end of session):")
        for k, v in sf_facts.items():
            print(f"  {k}: {v}")

        print("\nFull recall replies:")
        for name, r in results.items():
            print(f"\n  [{name}]\n{r['reply']}")

        # ── Assertions ────────────────────────────────────────────────────────

        # SF KV store must be non-empty — fact extraction must have worked
        assert len(sf_facts) > 0, "Sticky Facts KV store is empty — extraction never ran"

        # SF and BR must both recall at least as many facts as SW
        # (isolation/KV mechanisms must not make things worse)
        assert sf_r["score"] >= sw_r["score"], (
            f"Sticky Facts recall ({sf_r['score']}/4) should be ≥ SW ({sw_r['score']}/4)"
        )
        assert br_r["score"] >= sw_r["score"], (
            f"Branching ({br_r['score']}/4) should recall ≥ SW ({sw_r['score']}/4)"
        )

        # No strategy should score zero
        for name, r in results.items():
            assert r["score"] > 0, f"{name} recalled nothing — completely broken"

        # Main branch isolation: tangent messages did NOT grow the main branch
        assert len(br.current_branch.messages) == main_before + 2, (
            "Main branch should only have grown by 1 recall Q&A pair after returning"
        )

        # SF token overhead is bounded: extra fact-extraction calls < 4x SW total
        assert sf_r["total_tokens"] < sw_r["total_tokens"] * 4, (
            f"SF token overhead too high: {sf_r['total_tokens']} vs SW {sw_r['total_tokens']}"
        )


# ── Manual runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = _real_client()
    print("=" * 70)
    print("DATA MESH CASE — manual run (all 3 strategies)")
    print(f"Turns: {len(DATA_MESH_SCRIPT)} | Facts to recall: {list(KEY_FACTS.keys())}")
    print("=" * 70)

    strategies = [
        SlidingWindowStrategy(client, MODEL, SYSTEM_PROMPT, window_size=10),
        StickyFactsStrategy(client, MODEL, SYSTEM_PROMPT, window_size=6),
        BranchingStrategy(client, MODEL, SYSTEM_PROMPT, window_size=10),
    ]
    results = []
    for s in strategies:
        print(f"\n{'─'*50}")
        print(f"Running: {s.name}")
        print("─" * 50)
        results.append(_run_strategy(s, DATA_MESH_SCRIPT, delay=0.4))

    _print_results(results)
