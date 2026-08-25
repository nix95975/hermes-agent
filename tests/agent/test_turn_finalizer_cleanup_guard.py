"""Regression test for #8049.

When the post-loop cleanup chain in ``finalize_turn`` raises — trajectory
save (file I/O), resource teardown (remote VM/browser), or session
persistence (SQLite) — the partial ``final_response`` the caller is waiting
for must still be returned.  Previously any of those raised straight out of
``run_conversation``, so a subprocess wrapper saw an empty stdout with no
traceback and lost the whole turn.
"""

import pytest

from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, raise_in):
        self._raise_in = set(raise_in)
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    # --- fallible cleanup surfaces -------------------------------------
    def _save_trajectory(self, *a, **k):
        if "save_trajectory" in self._raise_in:
            raise RuntimeError("trajectory disk full")

    def _cleanup_task_resources(self, *a, **k):
        if "cleanup_task_resources" in self._raise_in:
            raise RuntimeError("docker teardown EOF")

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        if "persist_session" in self._raise_in:
            raise RuntimeError("sqlite database is locked")

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL SUMMARY FROM MODEL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _run(
    agent,
    *,
    final_response=None,
    api_call_count=3,
    turn_exit_reason="unknown",
):
    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason=turn_exit_reason,
    )




@pytest.mark.parametrize(
    "step", ["save_trajectory", "cleanup_task_resources", "persist_session"]
)
def test_single_cleanup_step_raises_does_not_skip_others(step):
    agent = _StubAgent(raise_in=(step,))
    result = _run(agent)
    # Response survives.
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    # Exactly the failing step is recorded; the others ran without error.
    assert result["cleanup_errors"] == [
        next(
            e
            for e in result["cleanup_errors"]
            if e.startswith(step)
        )
    ]
    assert len(result["cleanup_errors"]) == 1


def test_clean_turn_has_no_cleanup_errors_key():
    agent = _StubAgent(raise_in=())
    result = _run(agent)
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    assert result["completed"] is False
    assert "cleanup_errors" not in result


def test_turn_persistence_retry_and_callback_run_inside_finalizer():
    agent = _StubAgent(raise_in=())
    agent._turn_persist_retry_attempts = 2
    attempts = []
    callbacks = []

    def persist(*_args, **_kwargs):
        attempts.append("persist")
        if len(attempts) == 1:
            raise RuntimeError("transient lock")

    agent._persist_session = persist
    agent._turn_persistence_callback = lambda **kwargs: callbacks.append(kwargs)

    result = _run(
        agent,
        final_response="done",
        api_call_count=1,
        turn_exit_reason="text_response(done)",
    )

    assert attempts == ["persist", "persist"]
    assert callbacks == [{"succeeded": True, "successful_turn": True}]
    assert "cleanup_errors" not in result


def _real_persist_agent(*, flush_result, token_error=None):
    events = []

    class _DB:
        def flush_token_counts(self):
            events.append("token_flush")
            if token_error is not None:
                raise token_error

    agent = _StubAgent(raise_in=())
    agent._session_db = _DB()
    agent._session_persist_lock = None
    agent._persist_disabled = False
    agent.session_id = "persist-contract"
    agent._inflight_turn_id = "turn"
    agent._inflight_turn_session_id = "persist-contract"
    agent._drop_trailing_empty_response_scaffolding = lambda _messages: None
    agent._save_session_log = lambda _messages: events.append("json_log")
    agent._flush_messages_to_session_db = lambda *_args: (
        events.append("transcript_flush") or flush_result
    )
    agent._persist_session = AIAgent._persist_session.__get__(agent, AIAgent)
    return agent, events


def test_false_transcript_flush_retries_and_cannot_report_persisted():
    agent, events = _real_persist_agent(flush_result=False)
    agent._turn_persist_retry_attempts = 2
    callbacks = []
    agent._turn_persistence_callback = lambda **kwargs: callbacks.append(kwargs)

    result = _run(
        agent,
        final_response="done",
        api_call_count=1,
        turn_exit_reason="text_response(done)",
    )

    assert events == [
        "json_log",
        "transcript_flush",
        "json_log",
        "transcript_flush",
    ]
    assert callbacks == [{"succeeded": False, "successful_turn": True}]
    assert result["cleanup_errors"] == [
        "persist_session: Session transcript persistence failed"
    ]


def test_token_accounting_failure_does_not_relabel_committed_transcript():
    agent, events = _real_persist_agent(
        flush_result=True,
        token_error=RuntimeError("token accounting unavailable"),
    )
    callbacks = []
    agent._turn_persistence_callback = lambda **kwargs: callbacks.append(kwargs)

    result = _run(
        agent,
        final_response="done",
        api_call_count=1,
        turn_exit_reason="text_response(done)",
    )

    assert events == ["json_log", "transcript_flush", "token_flush"]
    assert callbacks == [{"succeeded": True, "successful_turn": True}]
    assert "cleanup_errors" not in result


def test_none_transcript_flush_remains_disabled_or_unavailable_not_failure():
    agent, events = _real_persist_agent(flush_result=None)

    agent._persist_session([], [])

    assert events == ["json_log", "transcript_flush", "token_flush"]


