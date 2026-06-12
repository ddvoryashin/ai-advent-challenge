"""
Quality comparison: SummaryAgent vs full-history (naive) agent.

Test scenario
─────────────
A scripted 15-turn conversation about a data pipeline project.
Four details are planted as scattered side-remarks across turns 3, 5, 7, 9
(NOT as a structured list in turn 1) — exactly the kind of information a
summarizer is likely to silently drop:

  • Cloud constraint (turn 3):  "GCP only, AWS is blocked by IT policy"
  • Broker switch (turn 5):     Kafka → Apache Pulsar
  • Latency SLA (turn 7):       200ms p99 (contractual obligation)
  • Team members (turn 9):      Alice (lead) + Bob (junior), exactly 2 people

All four end up inside the summarized chunks, not in the recent-5 window.
A summarizer focused on "architectural decisions" tends to drop these because
they look like side-notes embedded in Q&A, not main facts.

Turns 10-14 are filler. Turn 15 asks specifically about the four scattered details.

Metrics reported
────────────────
• Recall score (0-4): keyword hits in the final answer
• Request tokens on the last turn (actual, from API)
• Estimated full-history tokens (if everything were sent)
• Token savings of the summary approach
"""

import sys
import time
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from task9 import SummaryAgent, _estimate_tokens, _read_history, load_api_key, MODEL, SYSTEM_PROMPT

# ── Paths for test artefacts (cleaned up at the end) ─────────────────────────
_DIR = Path(__file__).parent
HISTORY_SUMMARY = _DIR / "_test_summary_history.jsonl"
SUMMARY_FILE = _DIR / "_test_summary.json"

# ── Details scattered across filler turns ────────────────────────────────────
# Planted as side-remarks, not structured facts — the summarizer is likely to lose them.
KEY_FACTS = {
    "cloud":    "GCP",     # turn 3: "GCP only, AWS blocked"
    "broker":   "Pulsar",  # turn 5: switched from Kafka to Apache Pulsar
    "latency":  "200ms",   # turn 7: contractual p99 SLA
    "lead_dev": "Alice",   # turn 9: team is Alice (lead) + Bob (junior)
}

# ── Scripted conversation ─────────────────────────────────────────────────────
SCRIPT: list[str] = [
    # Turn 1: generic setup — intentionally NO specific details here
    "We need to design a data pipeline for NovaTech. "
    "Budget is $120,000 and the deadline is September 2025. "
    "ClickHouse as the warehouse, 5 TB of event data daily.",

    # Turn 2: filler
    "What ingestion tool would you recommend for high-volume events?",

    # Turn 3: plant detail 1 — cloud constraint (looks like a passing remark)
    "Oh, and one more thing — IT policy blocks all AWS services. "
    "We can only run on GCP. I should have mentioned that upfront.",

    # Turn 4: filler
    "What are the trade-offs between batch and micro-batch ingestion at this scale?",

    # Turn 5: plant detail 2 — broker switch mid-conversation
    "Actually, forget Kafka — legal cleared Apache Pulsar last week, "
    "so we're switching to Pulsar for the event bus.",

    # Turn 6: filler
    "How should we partition ClickHouse tables for time-series event data?",

    # Turn 7: plant detail 3 — latency SLA buried in a follow-up
    "By the way, I checked the contract: there's a hard p99 latency SLA of 200ms "
    "for analytics dashboard queries. It's in the client agreement, so non-negotiable.",

    # Turn 8: filler
    "Should we use MergeTree or ReplacingMergeTree for the events table?",

    # Turn 9: plant detail 4 — team members (specific names)
    "Just to be clear on team size: we have exactly 2 engineers assigned. "
    "Alice is the tech lead, Bob is the junior. That's it — nobody else.",

    # Turns 10-14: filler — pushes planted details out of the 5-message window
    "How do we handle late-arriving events without corrupting aggregates?",
    "What monitoring stack would you set up for this pipeline?",
    "How do we handle schema evolution without downtime?",
    "What is a solid DR strategy for ClickHouse on GCP?",
    "How do we implement data quality checks at ingestion time?",

    # Turn 15: recall question — asks specifically about the four scattered details
    "Quick recap check: which cloud provider are we locked to, "
    "what message broker did we switch to, "
    "what is the contractual dashboard latency SLA, "
    "and who is the tech lead on the project?",
]

RECALL_TURN = SCRIPT[-1]
REQUIRED_KEYWORDS = list(KEY_FACTS.values())


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_recall(reply: str) -> tuple[int, list[str]]:
    """Return (score 0-4, list of found keywords)."""
    found = [kw for kw in REQUIRED_KEYWORDS if kw.lower() in reply.lower()]
    return len(found), found


class FullHistoryAgent:
    """Naive agent: sends the complete message history on every turn."""

    def __init__(self, client: OpenAI, model: str, system_prompt: str) -> None:
        self._client = client
        self._model = model
        self._system = {"role": "system", "content": system_prompt}
        self._history: list[dict] = []

    def chat(self, user_message: str) -> tuple[str, int, int]:
        """Returns (reply, request_tokens, estimated_full_history_tokens)."""
        self._history.append({"role": "user", "content": user_message})
        messages = [self._system] + self._history
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        reply = response.choices[0].message.content
        self._history.append({"role": "assistant", "content": reply})
        history_tokens = _estimate_tokens([self._system] + self._history)
        return reply, response.usage.prompt_tokens, history_tokens


def run_full_agent(client: OpenAI, script: list[str]) -> tuple[str, int, int]:
    agent = FullHistoryAgent(client, MODEL, SYSTEM_PROMPT)
    final_reply, last_req_tokens, last_hist_tokens = "", 0, 0
    for i, turn in enumerate(script):
        print(f"  full [{i + 1}/{len(script)}]...", end="\r", flush=True)
        final_reply, last_req_tokens, last_hist_tokens = agent.chat(turn)
        time.sleep(0.4)
    print()
    return final_reply, last_req_tokens, last_hist_tokens


def run_summary_agent(client: OpenAI, script: list[str]) -> tuple[str, int, int, dict]:
    agent = SummaryAgent(
        client, MODEL, SYSTEM_PROMPT,
        HISTORY_SUMMARY, SUMMARY_FILE,
        recent_keep=5, summary_chunk=10,
    )
    agent.reset()  # ensure clean state

    final_reply, last_req_tokens, last_hist_tokens = "", 0, 0
    for i, turn in enumerate(script):
        print(f"  summary [{i + 1}/{len(script)}]...", end="\r", flush=True)
        final_reply, stats = agent.chat(turn)
        last_req_tokens = stats.request_tokens
        last_hist_tokens = stats.history_tokens
        time.sleep(0.4)
    print()
    status = agent.status()
    agent.reset()  # clean up test files
    return final_reply, last_req_tokens, last_hist_tokens, status


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com")

    print("=" * 64)
    print("Quality test: Full history  vs  SummaryAgent (keep 5, chunk 10)")
    print(f"Turns: {len(SCRIPT)}  |  Facts to recall: {list(KEY_FACTS.values())}")
    print("=" * 64)

    # 1. Full-history agent
    print("\n[1/2] Running FULL-HISTORY agent …")
    full_reply, full_req, full_hist = run_full_agent(client, SCRIPT)
    full_score, full_found = score_recall(full_reply)
    print(f"\nFull-history recall answer:\n{full_reply}")
    print(f"\n  Score: {full_score}/4  |  Found: {full_found}")
    print(f"  Request tokens (last turn): {full_req}")
    print(f"  Estimated full-history tokens: {full_hist}")

    # 2. Summary agent
    print("\n" + "=" * 64)
    print("[2/2] Running SUMMARY agent …")
    sum_reply, sum_req, sum_hist, sum_status = run_summary_agent(client, SCRIPT)
    sum_score, sum_found = score_recall(sum_reply)
    print(f"\nSummary-agent recall answer:\n{sum_reply}")
    print(f"\n  Score: {sum_score}/4  |  Found: {sum_found}")
    print(f"  Request tokens (last turn): {sum_req}")
    print(f"  Estimated full-history tokens: {sum_hist}")
    print(f"  Agent status at end: {sum_status}")

    # 3. Comparison table
    savings = full_req - sum_req
    pct = round(100 * savings / full_req, 1) if full_req else 0
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"{'Metric':<38} {'Full':>10} {'Summary':>10}")
    print("-" * 60)
    print(f"{'Recall score (0-4)':<38} {full_score:>10} {sum_score:>10}")
    print(f"{'Keywords found':<38} {str(full_found):>10} {str(sum_found):>10}")
    print(f"{'Request tokens (last turn)':<38} {full_req:>10} {sum_req:>10}")
    print(f"{'Estimated full-history tokens':<38} {full_hist:>10} {sum_hist:>10}")
    print(f"{'Token savings on last request':<38} {'—':>10} {f'{savings} ({pct}%)':>10}")
    print("=" * 64)

    if sum_score == full_score:
        print("\nConclusion: summary preserved recall quality with fewer tokens.")
    elif sum_score < full_score:
        missing = [kw for kw in REQUIRED_KEYWORDS if kw not in sum_found]
        print(f"\nConclusion: summary lost facts: {missing} — summarizer needs tuning.")
    else:
        print("\nConclusion: summary agent outperformed full history (unexpected).")


if __name__ == "__main__":
    main()
