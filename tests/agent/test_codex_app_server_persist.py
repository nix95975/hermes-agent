"""Regression for #49225 — codex app-server turns must reach the session DB
exactly once.

The codex app-server runtime (``run_codex_app_server_turn``) is an early-return
path that bypasses ``conversation_loop`` and therefore must enter the shared
marker-aware turn-persistence seam itself. Successful projected output must be
persisted exactly once, while failed or interrupted projection must be dropped.

The inbound user turn is already flushed at turn start
(``turn_context._persist_session``), and a gateway raw re-write would duplicate
it (#860 / #42039). These tests lock in:

1. Successful Codex projection is persisted through the shared seam and the
   result returns ``agent_persisted=True``.
2. Failed, thrown, and interrupted turns persist only the safe inbound state.
3. The already-flushed user turn is not re-written, while successful projected
   assistant output lands once.
4. The gateway resolution expression preserves standard-runtime behaviour.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.codex_runtime import run_codex_app_server_turn
from hermes_state import SessionDB
from run_agent import AIAgent


def _make_turn():
    return SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[{"role": "assistant", "content": "CODEX_ASSISTANT"}],
        tool_iterations=0,
        final_text="CODEX_ASSISTANT",
        should_retire=False,
    )


def _make_agent(session_db=None, session_id="sess-codex"):
    agent = MagicMock()
    # Pre-seed the session so run_codex_app_server_turn skips the spawn block.
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = _make_turn()
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = session_db
    agent._session_db_created = True
    agent.session_id = session_id
    return agent


def _run(agent, messages=None):
    return run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=messages or [{"role": "user", "content": "hello"}],
        effective_task_id="task-1",
    )


def test_codex_success_persists_once_and_reports_callback_success():
    """A successful Codex turn uses the common persistence contract once."""
    agent = _make_agent(session_db=SimpleNamespace())
    callbacks = []
    agent._persist_session = MagicMock(return_value=None)
    agent._turn_persistence_callback = lambda **kwargs: callbacks.append(kwargs)

    result = _run(agent)

    assert result["completed"] is True
    assert result.get("failed") is not True
    assert result["agent_persisted"] is True
    agent._persist_session.assert_called_once()
    persisted_messages = agent._persist_session.call_args.args[0]
    assert [message["content"] for message in persisted_messages] == [
        "hello",
        "CODEX_ASSISTANT",
    ]
    assert callbacks == [{"succeeded": True, "successful_turn": True}]


def test_codex_no_db_keeps_success_and_reports_callback_success():
    """No-DB persistence remains a successful no-op, as on the normal path."""
    agent = _make_agent(session_db=None)
    callbacks = []
    agent._persist_session = MagicMock(return_value=None)
    agent._turn_persistence_callback = lambda **kwargs: callbacks.append(kwargs)

    result = _run(agent)

    assert result["completed"] is True
    assert isinstance(result["messages"][-1]["timestamp"], float)
    assert result["agent_persisted"] is True
    assert callbacks == [{"succeeded": True, "successful_turn": True}]


def test_codex_false_flush_retries_then_fails_closed_with_callback():
    """An authoritative False flush is retried and cannot report completion."""
    agent = _make_agent(session_db=SimpleNamespace(flush_token_counts=lambda: None))
    agent._turn_persist_retry_attempts = 2
    agent._session_persist_lock = None
    agent._persist_disabled = False
    agent._inflight_turn_id = "turn-1"
    agent._inflight_turn_session_id = agent.session_id
    agent._drop_trailing_empty_response_scaffolding = MagicMock()
    agent._save_session_log = MagicMock()
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    agent._persist_session = AIAgent._persist_session.__get__(agent, AIAgent)
    callbacks = []
    agent._turn_persistence_callback = lambda **kwargs: callbacks.append(kwargs)

    result = _run(agent)

    assert agent._flush_messages_to_session_db.call_count == 2
    assert callbacks == [{"succeeded": False, "successful_turn": True}]
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["agent_persisted"] is True
    assert result["error"]
    assert result["failure_reason"].startswith("session_persistence_failed:")
    assert [message["content"] for message in result["messages"]] == [
        "hello",
        "CODEX_ASSISTANT",
    ]


def test_codex_user_interrupt_is_reported_and_cleared():
    agent = _make_agent(session_db=None)
    turn = _make_turn()
    turn.interrupted = True
    turn.final_text = ""
    agent._codex_session.run_turn.return_value = turn
    agent._interrupt_requested = True
    agent._interrupt_message = "new correction"

    def clear_interrupt():
        agent._interrupt_requested = False
        agent._interrupt_message = None

    agent.clear_interrupt.side_effect = clear_interrupt
    result = run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="task-1",
    )

    assert result["interrupted"] is True
    assert result["interrupt_message"] == "new correction"
    agent.clear_interrupt.assert_called_once_with()
    assert agent._interrupt_requested is False


def test_codex_turn_persists_each_message_exactly_once():
    """The user turn (flushed at turn start) must not be duplicated; the
    projected assistant message must land once.  Uses a real SessionDB and the
    real AIAgent._flush_messages_to_session_db to prove no #860/#42039
    duplicate-write regression on the codex path."""
    tmp = tempfile.mkdtemp(prefix="codex_persist_")
    try:
        db = SessionDB(Path(tmp) / "state.db")
        sid = "sess-codex-once"
        db.create_session(session_id=sid, source="telegram", model="codex")

        # Real agent bound to this DB/session, minimal construction.
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True
        agent._codex_session = MagicMock()
        agent._codex_session.run_turn.return_value = _make_turn()
        agent.tool_progress_callback = None

        # Model the real flow: the inbound user turn is flushed at turn start
        # (turn_context._persist_session) on the SAME `messages` list the codex
        # path later reuses. That flush stamps _DB_PERSISTED_MARKER on the user
        # dict, so the codex-path flush skips it — no duplicate.
        user_msg = {"role": "user", "content": "USER_TURN"}
        messages = [user_msg]
        agent._flush_messages_to_session_db(messages)  # turn-start flush

        result = run_codex_app_server_turn(
            agent,
            user_message="USER_TURN",
            original_user_message="USER_TURN",
            messages=messages,
            effective_task_id="task-1",
        )
        assert result["agent_persisted"] is True

        rows = db.get_messages(sid, include_inactive=True)
        contents = [r["content"] for r in rows]
        # Exactly one user turn, exactly one assistant turn — no duplicates.
        assert contents.count("USER_TURN") == 1, contents
        assert contents.count("CODEX_ASSISTANT") == 1, contents
        assistant_row = next(
            row for row in rows if row["content"] == "CODEX_ASSISTANT"
        )
        assert isinstance(assistant_row["timestamp"], float)
        # session_search can now see the codex conversation.
        hits = {r["session_id"] for r in db.search_messages("CODEX_ASSISTANT")}
        assert sid in hits
    finally:
        import shutil

        shutil.rmtree(tmp)


def _make_real_db_agent(tmp_path, session_id):
    db = SessionDB(tmp_path / f"{session_id}.db")
    db.create_session(session_id=session_id, source="api_server", model="codex")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=db,
        session_id=session_id,
    )
    agent._session_db_created = True
    agent._codex_session = MagicMock()
    agent.tool_progress_callback = None
    user_message = {"role": "user", "content": "DURABLE_USER"}
    messages = [user_message]
    agent._flush_messages_to_session_db(messages)
    return db, agent, messages


def test_codex_returned_error_discards_projected_output_in_real_db(tmp_path):
    db, agent, messages = _make_real_db_agent(tmp_path, "returned-error")
    callbacks = []
    agent._turn_persistence_callback = lambda **kwargs: callbacks.append(kwargs)
    turn = _make_turn()
    turn.error = "codex timed out"
    turn.final_text = "FAILED_PARTIAL"
    turn.projected_messages = [
        {
            "role": "assistant",
            "content": "FAILED_PARTIAL",
            "tool_calls": [{
                "id": "failed-call",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "failed-call", "content": "FAILED_TOOL"},
    ]
    agent._codex_session.run_turn.return_value = turn

    result = _run(agent, messages)

    assert result["completed"] is False
    assert result["partial"] is True
    assert result["failed"] is True
    assert result["error"] == "codex timed out"
    assert result["failure_reason"] == "codex_turn_error"
    assert result["agent_persisted"] is True
    assert [message["content"] for message in result["messages"]] == ["DURABLE_USER"]
    assert callbacks == [{"succeeded": True, "successful_turn": False}]
    assert [row["content"] for row in db.get_messages("returned-error")] == [
        "DURABLE_USER"
    ]
    db.close()


def test_codex_thrown_exception_persists_safe_state_once_in_real_db(tmp_path):
    db, agent, messages = _make_real_db_agent(tmp_path, "thrown-error")
    callbacks = []
    agent._turn_persistence_callback = lambda **kwargs: callbacks.append(kwargs)
    agent._codex_session.run_turn.side_effect = RuntimeError("codex crashed")

    result = _run(agent, messages)

    assert result["completed"] is False
    assert result["partial"] is True
    assert result["failed"] is True
    assert result["error"] == "codex crashed"
    assert result["failure_reason"] == "codex_turn_exception"
    assert result["agent_persisted"] is True
    assert callbacks == [{"succeeded": True, "successful_turn": False}]
    assert [row["content"] for row in db.get_messages("thrown-error")] == [
        "DURABLE_USER"
    ]
    db.close()


def test_codex_interruption_discards_projected_output_in_real_db(tmp_path):
    db, agent, messages = _make_real_db_agent(tmp_path, "interrupted-turn")
    callbacks = []
    agent._turn_persistence_callback = lambda **kwargs: callbacks.append(kwargs)
    turn = _make_turn()
    turn.interrupted = True
    turn.final_text = "INTERRUPTED_PARTIAL"
    turn.projected_messages = [
        {"role": "assistant", "content": "INTERRUPTED_PARTIAL"}
    ]
    agent._codex_session.run_turn.return_value = turn
    agent._interrupt_requested = True
    agent._interrupt_message = "stop now"

    result = _run(agent, messages)

    assert result["completed"] is False
    assert result["partial"] is True
    assert result["interrupted"] is True
    assert result["interrupt_message"] == "stop now"
    assert result.get("failed") is not True
    assert result["agent_persisted"] is True
    assert [message["content"] for message in result["messages"]] == ["DURABLE_USER"]
    assert callbacks == [{"succeeded": True, "successful_turn": False}]
    assert [row["content"] for row in db.get_messages("interrupted-turn")] == [
        "DURABLE_USER"
    ]
    db.close()


class TestGatewayPersistedResolution:
    """The gateway default must preserve standard-runtime skip-db behaviour."""

    @staticmethod
    def _resolve_persistence_block(agent_result, session_db_present):
        # gateway/run.py persistence block:
        #   agent_persisted = agent_result.get("agent_persisted", self._session_db is not None)
        return agent_result.get("agent_persisted", session_db_present)

    @staticmethod
    def _resolve_passthrough(result_holder0):
        # gateway/run.py result_holder passthrough:
        #   result_holder[0].get("agent_persisted", True) if result_holder[0] else True
        return result_holder0.get("agent_persisted", True) if result_holder0 else True

    def test_codex_result_keeps_gateway_skip(self):
        # Codex now self-persists → gateway must SKIP (agent_persisted True).
        codex = {"agent_persisted": True}
        assert self._resolve_persistence_block(codex, True) is True
        assert self._resolve_persistence_block(codex, False) is True
        assert self._resolve_passthrough(codex) is True

    def test_standard_runtime_preserves_skip_db(self):
        # Standard runtime omits the key → old behaviour: skip iff DB present.
        standard = {"final_response": "ok"}
        assert self._resolve_persistence_block(standard, True) is True
        assert self._resolve_persistence_block(standard, False) is False
        assert self._resolve_passthrough(standard) is True

    def test_missing_result_holder_defaults_persisted(self):
        assert self._resolve_passthrough(None) is True
