"""Tests for /v1/runs endpoints: start, status, events, steer, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/steer — inject guidance into a running agent
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import json
import threading
import time
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _approval_event_choices,
    cors_middleware,
    security_headers_middleware,
)
from hermes_state import SessionDB
from run_agent import AIAgent
from tools import approval as approval_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_session", "allow_permanent", "expected"),
    [
        (False, True, True, ["once", "session", "always", "deny"]),
        (False, True, False, ["once", "session", "deny"]),
        (False, False, True, ["once", "deny"]),
        (False, False, False, ["once", "deny"]),
        (True, True, True, ["once", "deny"]),
        (True, False, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_session, allow_permanent, expected
):
    assert (
        _approval_event_choices(
            smart_denied=smart_denied,
            allow_session=allow_session,
            allow_permanent=allow_permanent,
        )
        == expected
    )


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    return app


def _terminal_events(body: str) -> list[dict]:
    events = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    return [event for event in events if event.get("event") in {
        "run.completed", "run.failed", "run.cancelled"
    }]


async def _subscribe_to_run(cli: TestClient, run_id: str, *, headers=None):
    response = await cli.get(f"/v1/runs/{run_id}/events", headers=headers)
    assert response.status == 200
    return _terminal_events(await response.text())


class _DeterministicPersistingAgent:
    """Model stub using AIAgent's real transcript persistence seam."""

    def __init__(self, db: SessionDB, session_id: str, seen_histories: list):
        self._session_db = db
        self.session_id = session_id
        self.seen_histories = seen_histories
        self._persist_disabled = False
        self._session_db_created = True
        self._session_persist_lock = None
        self._last_flushed_db_idx = 0
        self._flushed_db_message_session_id = None
        self._flushed_db_message_ids = set()
        self._db_flush_scan_prefix = None
        self._persist_user_message_idx = None
        self._persist_user_message_override = None
        self._persist_user_message_timestamp = None
        self._pending_cli_user_message = None
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self._flush_messages_to_session_db_unlocked = MethodType(
            AIAgent._flush_messages_to_session_db_unlocked, self
        )
        self._persist_session = MethodType(
            AIAgent._flush_messages_to_session_db, self
        )

    def run_conversation(self, user_message=None, conversation_history=None, task_id=None):
        history = list(conversation_history or [])
        self.seen_histories.append(history)
        turn_number = len(self.seen_histories)
        messages = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"answer {turn_number}"},
        ]
        self._persist_session(messages, history)
        return {"final_response": f"answer {turn_number}", "messages": messages}


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


def _make_production_codex_agent(db: SessionDB, session_id: str) -> AIAgent:
    """Real AIAgent conversation/persistence path with only Codex I/O stubbed."""
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="codex_app_server",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=db,
        session_id=session_id,
    )
    agent._session_db_created = True
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-api",
        turn_id="turn-api",
        projected_messages=[
            {"role": "assistant", "content": "CODEX_API_ASSISTANT"}
        ],
        tool_iterations=0,
        final_text="CODEX_API_ASSISTANT",
        should_retire=False,
    )
    agent.tool_progress_callback = None
    return agent


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter(tmp_path):
    adapter = _make_adapter(api_key="sk-secret")
    db = SessionDB(tmp_path / "auth-state.db")
    adapter._session_db = db
    try:
        yield adapter
    finally:
        db.close()


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_binds_chat_id_for_delegation_wake_target(self, auth_adapter):
        """/v1/runs must bind the raw session id as the api_server chat_id
        (like every other agent-entry route does via _run_agent): the async
        delegation dispatch reads HERMES_SESSION_CHAT_ID to pick its wake
        self-post target, and an empty binding forces background delegations
        on this route back to synchronous execution."""
        auth_adapter._session_db.create_session("runs-raw-sid", "api_server")
        app = _create_runs_app(auth_adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(user_message=None, conversation_history=None, task_id=None):
                    from tools.async_delegation import _current_origin_session_id

                    captured["origin_session_id"] = _current_origin_session_id()
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "runs-raw-sid"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(
                        f"/v1/runs/{run_id}",
                        headers={"Authorization": "Bearer sk-secret"},
                    )
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured.get("origin_session_id") == "runs-raw-sid", (
            "runs route must bind chat_id so delegation dispatch sees a wake target"
        )

    @pytest.mark.asyncio
    async def test_start_with_session_id_loads_persisted_history(self, auth_adapter, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        auth_adapter._session_db = db
        session_id = db.create_session("continued-session", "api_server")
        db.append_message(session_id, "user", "earlier question")
        db.append_message(session_id, "assistant", "earlier answer")

        app = _create_runs_app(auth_adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(auth_adapter, "_create_agent") as mock_create:
                    mock_agent = MagicMock()
                    mock_agent.run_conversation.return_value = {"final_response": "done"}
                    mock_agent.session_prompt_tokens = 0
                    mock_agent.session_completion_tokens = 0
                    mock_agent.session_total_tokens = 0
                    mock_create.return_value = mock_agent

                    resp = await cli.post(
                        "/v1/runs",
                        json={"input": "follow-up", "session_id": session_id},
                        headers={"Authorization": "Bearer sk-secret"},
                    )
                    assert resp.status == 202
                    await resp.json()
                    for _ in range(20):
                        if mock_agent.run_conversation.called:
                            break
                        await asyncio.sleep(0.01)

                    assert mock_agent.run_conversation.called
                    loaded_history = mock_agent.run_conversation.call_args.kwargs[
                        "conversation_history"
                    ]
                    assert [
                        {"role": message["role"], "content": message["content"]}
                        for message in loaded_history
                    ] == [
                        {"role": "user", "content": "earlier question"},
                        {"role": "assistant", "content": "earlier answer"},
                    ]
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_session_history_fallback_requires_configured_api_key(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "follow-up", "session_id": "private-session"},
            )

        assert resp.status == 403
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_body_session_id_auth_fails_before_session_lookup(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        with patch.object(
            auth_adapter,
            "_get_existing_session_or_404",
            new_callable=AsyncMock,
        ) as lookup:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "follow-up", "session_id": "private-session"},
                )

        assert response.status == 401
        lookup.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "session_id",
        [
            "",
            None,
            "../../private",
            "/absolute",
            "..\\windows",
            "bad\x00id",
            123,
            "x" * 257,
        ],
    )
    async def test_session_history_fallback_rejects_invalid_id(
        self, auth_adapter, session_id
    ):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "follow-up", "session_id": session_id},
                headers={"Authorization": "Bearer sk-secret"},
            )

        assert resp.status == 400
        assert auth_adapter._run_streams == {}
        assert auth_adapter._run_statuses == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "explicit_history",
        [
            [{"role": "user", "content": "client-provided history"}],
            [],
        ],
        ids=["populated", "empty"],
    )
    async def test_explicit_history_takes_precedence_over_session_db(
        self, auth_adapter, explicit_history
    ):
        auth_adapter._session_db.create_session("continued-session", "api_server")
        app = _create_runs_app(auth_adapter)
        with (
            patch.object(
                auth_adapter,
                "_conversation_history_for_session",
                new_callable=AsyncMock,
            ) as mock_load_history,
            patch.object(auth_adapter, "_create_agent") as mock_create,
        ):
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "done"}
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            mock_create.return_value = mock_agent

            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "follow-up",
                        "session_id": "continued-session",
                        "conversation_history": explicit_history,
                    },
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 202
                await resp.json()
                for _ in range(20):
                    if mock_agent.run_conversation.called:
                        break
                    await asyncio.sleep(0.01)

        assert mock_agent.run_conversation.called
        mock_load_history.assert_not_awaited()
        assert mock_agent.run_conversation.call_args.kwargs[
            "conversation_history"
        ] == explicit_history

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body, expected_history",
        [
            ({"previous_response_id": "resp_unknown"}, []),
            ({"previous_response_id": ""}, []),
            ({"input": [
                {"role": "user", "content": "legacy earlier"},
                {"role": "user", "content": "follow-up"},
            ]}, [{"role": "user", "content": "legacy earlier"}]),
        ],
        ids=["unknown-previous-response", "empty-previous-response", "legacy-input"],
    )
    async def test_caller_history_source_presence_prevents_session_db_fallback(
        self, auth_adapter, tmp_path, body, expected_history
    ):
        db = SessionDB(tmp_path / "state.db")
        auth_adapter._session_db = db
        session_id = db.create_session("source-precedence", "api_server")
        db.append_message(session_id, "user", "private persisted history")
        payload = {"input": "follow-up", "session_id": session_id, **body}

        app = _create_runs_app(auth_adapter)
        try:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                agent = MagicMock()
                preserve_flags = []

                def run_with_caller_history(**_kwargs):
                    preserve_flags.append(
                        agent._preserve_caller_history_after_lease_wait
                    )
                    return {"final_response": "done"}

                agent.run_conversation.side_effect = run_with_caller_history
                agent.session_prompt_tokens = 0
                agent.session_completion_tokens = 0
                agent.session_total_tokens = 0
                mock_create.return_value = agent
                async with TestClient(TestServer(app)) as cli:
                    response = await cli.post(
                        "/v1/runs",
                        json=payload,
                        headers={"Authorization": "Bearer sk-secret"},
                    )
                    assert response.status == 202
                    run_id = (await response.json())["run_id"]
                    terminals = await _subscribe_to_run(
                        cli, run_id, headers={"Authorization": "Bearer sk-secret"}
                    )

            assert [event["event"] for event in terminals] == ["run.completed"]
            assert (
                agent.run_conversation.call_args.kwargs["conversation_history"]
                == expected_history
            )
            assert preserve_flags == [True]
            assert "_preserve_caller_history_after_lease_wait" not in agent.__dict__
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_missing_body_session_id_returns_404_before_run_admission(
        self, auth_adapter, tmp_path
    ):
        db = SessionDB(tmp_path / "state.db")
        auth_adapter._session_db = db
        app = _create_runs_app(auth_adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "follow-up", "session_id": "missing-session"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                payload = await response.json()

            assert response.status == 404
            assert payload["error"]["code"] == "session_not_found"
            assert auth_adapter._run_streams == {}
            assert auth_adapter._run_statuses == {}
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_session_history_load_failure_admits_no_run(self, auth_adapter):
        auth_adapter._session_db.create_session("broken-history", "api_server")
        app = _create_runs_app(auth_adapter)
        with (
            patch.object(
                auth_adapter._session_db,
                "get_messages_as_conversation",
                side_effect=RuntimeError("database disk image is malformed"),
            ),
            patch.object(auth_adapter, "_create_agent") as create_agent,
        ):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "follow-up", "session_id": "broken-history"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                payload = await response.json()

        assert response.status == 503
        assert payload["error"]["code"] == "session_db_unavailable"
        create_agent.assert_not_called()
        assert auth_adapter._run_streams == {}
        assert auth_adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_session_lookup_failure_admits_no_run(self, auth_adapter):
        auth_adapter._session_db.create_session("broken-lookup", "api_server")
        app = _create_runs_app(auth_adapter)
        with (
            patch.object(
                auth_adapter._session_db,
                "get_session",
                side_effect=RuntimeError("Cannot operate on a closed database"),
            ),
            patch.object(auth_adapter, "_create_agent") as create_agent,
        ):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "follow-up", "session_id": "broken-lookup"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                payload = await response.json()

        assert response.status == 503
        assert payload["error"]["code"] == "session_db_unavailable"
        create_agent.assert_not_called()
        assert auth_adapter._run_streams == {}
        assert auth_adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_success_persists_authoritative_turn_messages(self, auth_adapter, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        auth_adapter._session_db = db
        session_id = db.create_session("persist-success", "api_server")
        messages = [
            {
                "role": "user",
                "content": "[CONTEXT SUMMARY]: earlier compacted context",
                "_compressed_summary": True,
            },
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "authoritative answer"},
        ]
        agent = _DeterministicPersistingAgent(db, session_id, [])

        def _run_with_tools(_self, **_kwargs):
            _self._persist_session(messages, [])
            _self._persist_session(messages, [])
            return {
                "final_response": "authoritative answer",
                "messages": messages,
            }

        agent.run_conversation = MethodType(_run_with_tools, agent)

        app = _create_runs_app(auth_adapter)
        try:
            with patch.object(auth_adapter, "_create_agent", return_value=agent):
                async with TestClient(TestServer(app)) as cli:
                    response = await cli.post(
                        "/v1/runs",
                        json={"input": "inspect", "session_id": session_id},
                        headers={"Authorization": "Bearer sk-secret"},
                    )
                    run_id = (await response.json())["run_id"]
                    terminals = await _subscribe_to_run(
                        cli, run_id, headers={"Authorization": "Bearer sk-secret"}
                    )

            assert [event["event"] for event in terminals] == ["run.completed"]
            loaded = db.get_messages_as_conversation(session_id)
            stored_rows = db.get_messages(session_id)
            assert len(loaded) == len(messages)
            assert stored_rows[0]["_compressed_summary"] is True
            assert loaded[0]["display_kind"] == "hidden"
            assert loaded[2]["tool_calls"] == messages[2]["tool_calls"]
            assert loaded[3]["tool_call_id"] == "call_1"
            assert loaded[3]["content"] == "result"
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_session_owned_history_survives_adapter_restart(self, tmp_path):
        state_path = tmp_path / "state.db"
        headers = {"Authorization": "Bearer sk-secret"}
        seen_histories = []

        adapter1 = _make_adapter(api_key="sk-secret")
        db1 = SessionDB(state_path)
        adapter1._session_db = db1
        app1 = _create_runs_app(adapter1)
        async with TestClient(TestServer(app1)) as cli:
            create = await cli.post(
                "/api/sessions", json={"title": "runs continuity"}, headers=headers
            )
            assert create.status == 201
            session_id = (await create.json())["session"]["id"]
            with patch.object(
                adapter1,
                "_create_agent",
                side_effect=lambda **kwargs: _DeterministicPersistingAgent(
                    db1, kwargs["session_id"], seen_histories
                ),
            ):
                first = await cli.post(
                    "/v1/runs",
                    json={"input": "turn one", "session_id": session_id},
                    headers=headers,
                )
                assert first.status == 202
                first_run_id = (await first.json())["run_id"]
                first_terminals = await _subscribe_to_run(
                    cli, first_run_id, headers=headers
                )

        assert [event["event"] for event in first_terminals] == ["run.completed"]
        await adapter1.disconnect()
        db1.close()
        adapter1._session_db = None

        adapter2 = _make_adapter(api_key="sk-secret")
        db2 = SessionDB(state_path)
        adapter2._session_db = db2
        app2 = _create_runs_app(adapter2)
        try:
            with patch.object(
                adapter2,
                "_create_agent",
                side_effect=lambda **kwargs: _DeterministicPersistingAgent(
                    db2, kwargs["session_id"], seen_histories
                ),
            ):
                async with TestClient(TestServer(app2)) as cli:
                    second = await cli.post(
                        "/v1/runs",
                        json={"input": "turn two", "session_id": session_id},
                        headers=headers,
                    )
                    assert second.status == 202
                    second_run_id = (await second.json())["run_id"]
                    second_terminals = await _subscribe_to_run(
                        cli, second_run_id, headers=headers
                    )

            assert [event["event"] for event in second_terminals] == ["run.completed"]
            assert [
                {"role": message["role"], "content": message["content"]}
                for message in seen_histories[1]
            ] == [
                {"role": "user", "content": "turn one"},
                {"role": "assistant", "content": "answer 1"},
            ]
            assert [
                {"role": message["role"], "content": message["content"]}
                for message in db2.get_messages_as_conversation(session_id)
            ] == [
                {"role": "user", "content": "turn one"},
                {"role": "assistant", "content": "answer 1"},
                {"role": "user", "content": "turn two"},
                {"role": "assistant", "content": "answer 2"},
            ]
        finally:
            await adapter2.disconnect()
            db2.close()

    @pytest.mark.asyncio
    async def test_busy_canonical_session_fails_fast_then_retry_succeeds(self, tmp_path):
        state_path = tmp_path / "shared-state.db"
        db1 = SessionDB(state_path)
        session_id = db1.create_session("shared-process-session", "api_server")
        db2 = SessionDB(state_path)
        adapter1 = _make_adapter(api_key="sk-secret")
        adapter2 = _make_adapter(api_key="sk-secret")
        adapter1._session_db = db1
        adapter2._session_db = db2
        headers = {"Authorization": "Bearer sk-secret"}
        first_started = threading.Event()
        release_first = threading.Event()
        seen_histories = []
        first_agent = _DeterministicPersistingAgent(db1, session_id, seen_histories)
        second_agent = _DeterministicPersistingAgent(db2, session_id, seen_histories)
        first_run = first_agent.run_conversation

        def _blocked_first(_self, **kwargs):
            first_started.set()
            release_first.wait(timeout=5)
            return first_run(**kwargs)

        first_agent.run_conversation = MethodType(_blocked_first, first_agent)
        app1 = _create_runs_app(adapter1)
        app2 = _create_runs_app(adapter2)
        try:
            with (
                patch.object(adapter1, "_create_agent", return_value=first_agent),
                patch.object(adapter2, "_create_agent", return_value=second_agent) as create2,
            ):
                async with (
                    TestClient(TestServer(app1)) as cli1,
                    TestClient(TestServer(app2)) as cli2,
                ):
                    response1 = await cli1.post(
                        "/v1/runs",
                        json={"input": "turn one", "session_id": session_id},
                        headers=headers,
                    )
                    run1 = (await response1.json())["run_id"]
                    assert first_started.wait(timeout=3)

                    response2 = await asyncio.wait_for(
                        cli2.post(
                            "/v1/runs",
                            json={"input": "turn two", "session_id": session_id},
                            headers=headers,
                        ),
                        timeout=2.0,
                    )
                    busy = await response2.json()
                    assert response2.status == 409
                    assert busy["error"]["code"] == "session_turn_lease_busy"
                    create2.assert_not_called()
                    assert adapter2._run_streams == {}
                    assert adapter2._run_statuses == {}

                    other_id = db2.create_session("unrelated", "api_server")
                    unrelated = await cli2.post(
                        "/v1/runs",
                        json={"input": "other", "session_id": other_id},
                        headers=headers,
                    )
                    assert unrelated.status == 202
                    unrelated_run = (await unrelated.json())["run_id"]
                    await _subscribe_to_run(cli2, unrelated_run, headers=headers)

                    release_first.set()
                    await _subscribe_to_run(cli1, run1, headers=headers)
                    retry = await cli2.post(
                        "/v1/runs",
                        json={"input": "turn two", "session_id": session_id},
                        headers=headers,
                    )
                    assert retry.status == 202
                    run2 = (await retry.json())["run_id"]
                    await _subscribe_to_run(cli2, run2, headers=headers)

            assert seen_histories[0] == []
            assert [
                (message["role"], message["content"])
                for message in seen_histories[2]
            ] == [("user", "turn one"), ("assistant", "answer 2")]
        finally:
            await adapter1.disconnect()
            await adapter2.disconnect()
            db1.close()
            db2.close()

    @pytest.mark.asyncio
    async def test_cancelled_preflight_releases_exact_cached_profile_db_lease(
        self, tmp_path, monkeypatch
    ):
        adapter = _make_adapter(api_key="sk-secret")
        home = tmp_path / "profile-home"
        home.mkdir()
        db = SessionDB(home / "state.db")
        session_id = db.create_session("cancel-preflight", "api_server")
        adapter._session_db = None
        adapter._session_dbs[str(home)] = db
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: home
        )
        acquired = threading.Event()
        release_acquire = threading.Event()
        original_acquire = db.acquire_session_turn_lease

        def delayed_acquire(*args, **kwargs):
            result = original_acquire(*args, **kwargs)
            acquired.set()
            release_acquire.wait(timeout=5)
            return result

        monkeypatch.setattr(db, "acquire_session_turn_lease", delayed_acquire)
        task = asyncio.create_task(
            adapter._conversation_history_for_runs_session(session_id)
        )
        assert await asyncio.to_thread(acquired.wait, 3)
        task.cancel()
        release_acquire.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert db.try_acquire_session_turn_lease(
            session_id, "pid=retry:turn=next", ttl_seconds=30
        )
        db.release_session_turn_lease(session_id, "pid=retry:turn=next")
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_prestart_cancel_releases_exact_cached_profile_db_lease(
        self, tmp_path, monkeypatch
    ):
        adapter = _make_adapter(api_key="sk-secret")
        home = tmp_path / "profile-home"
        home.mkdir()
        db = SessionDB(home / "state.db")
        session_id = db.create_session("prestart-cancel", "api_server")
        adapter._session_db = None
        adapter._session_dbs[str(home)] = db
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
        original_set_status = adapter._set_run_status

        def cancel_when_queued(run_id, status, **kwargs):
            original_set_status(run_id, status, **kwargs)
            if status == "queued":
                adapter._stopping_run_ids.add(run_id)

        monkeypatch.setattr(adapter, "_set_run_status", cancel_when_queued)
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent") as create_agent:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "follow-up", "session_id": session_id},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                run_id = (await response.json())["run_id"]
                terminals = await _subscribe_to_run(
                    cli,
                    run_id,
                    headers={"Authorization": "Bearer sk-secret"},
                )

        assert [event["event"] for event in terminals] == ["run.cancelled"]
        create_agent.assert_not_called()
        assert db.try_acquire_session_turn_lease(
            session_id, "pid=retry:turn=next", ttl_seconds=30
        )
        db.release_session_turn_lease(session_id, "pid=retry:turn=next")
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_create_agent_failure_releases_exact_cached_profile_db_lease(
        self, tmp_path, monkeypatch
    ):
        adapter = _make_adapter(api_key="sk-secret")
        home = tmp_path / "profile-home"
        home.mkdir()
        db = SessionDB(home / "state.db")
        session_id = db.create_session("create-failure", "api_server")
        adapter._session_db = None
        adapter._session_dbs[str(home)] = db
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: home
        )
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent", side_effect=RuntimeError("boom")):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "follow-up", "session_id": session_id},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                run_id = (await response.json())["run_id"]
                terminals = await _subscribe_to_run(
                    cli,
                    run_id,
                    headers={"Authorization": "Bearer sk-secret"},
                )

        assert [event["event"] for event in terminals] == ["run.failed"]
        assert db.try_acquire_session_turn_lease(
            session_id, "pid=retry:turn=next", ttl_seconds=30
        )
        db.release_session_turn_lease(session_id, "pid=retry:turn=next")
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_start_rejects_conflicting_route_and_request_provider(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "model_routes": {
                        "alias": {
                            "model": "route/model",
                            "provider": "openrouter",
                        }
                    }
                },
            )
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "alias",
                        "provider": "minimax",
                    },
                )
                data = await resp.json()

        assert resp.status == 400
        assert "provider" in data["error"]["message"].lower()
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_passes_request_model_provider_options_to_create_agent(self, adapter):
        app = _create_runs_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 202
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, auth_adapter):
        auth_adapter._session_db.create_session("space-session", "api_server")
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(
                        f"/v1/runs/{run_id}",
                        headers={"Authorization": "Bearer sk-secret"},
                    )
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body
                assert [event["event"] for event in _terminal_events(body)] == [
                    "run.completed"
                ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "outcome, expected",
        [
            ({"failed": True, "error": "structured"}, "run.failed"),
            (RuntimeError("exploded"), "run.failed"),
        ],
        ids=["structured-failure", "exception"],
    )
    async def test_failure_paths_emit_exactly_one_terminal(
        self, adapter, outcome, expected
    ):
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent") as mock_create:
            agent = MagicMock()
            if isinstance(outcome, Exception):
                agent.run_conversation.side_effect = outcome
            else:
                agent.run_conversation.return_value = outcome
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            mock_create.return_value = agent

            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await response.json())["run_id"]
                terminals = await _subscribe_to_run(cli, run_id)

        assert [event["event"] for event in terminals] == [expected]
        agent._persist_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_transcript_persistence_failure_cannot_emit_completed(self, adapter):
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent") as mock_create:
            agent = MagicMock()

            def _run(*_args, **_kwargs):
                agent._turn_persistence_callback(
                    succeeded=False, successful_turn=True
                )
                return {"final_response": "not durable"}

            agent.run_conversation.side_effect = _run
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            mock_create.return_value = agent

            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await response.json())["run_id"]
                terminals = await _subscribe_to_run(cli, run_id)

        assert [event["event"] for event in terminals] == ["run.failed"]
        assert "_turn_persistence_callback" not in agent.__dict__

    @pytest.mark.asyncio
    async def test_canonical_codex_persistence_failure_is_pollable_failed(
        self, auth_adapter
    ):
        session_id = auth_adapter._session_db.create_session(
            "codex-persist-failure", "api_server"
        )
        agent = _make_production_codex_agent(auth_adapter._session_db, session_id)
        real_flush = agent._flush_messages_to_session_db
        assistant_attempts = 0

        def _fail_assistant_flush(messages, conversation_history=None):
            nonlocal assistant_attempts
            if any(
                message.get("content") == "CODEX_API_ASSISTANT"
                for message in messages
                if isinstance(message, dict)
            ):
                assistant_attempts += 1
                return False
            return real_flush(messages, conversation_history)

        agent._flush_messages_to_session_db = _fail_assistant_flush
        app = _create_runs_app(auth_adapter)
        headers = {"Authorization": "Bearer sk-secret"}

        with patch.object(auth_adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": session_id},
                    headers=headers,
                )
                assert response.status == 202
                run_id = (await response.json())["run_id"]
                terminals = await _subscribe_to_run(cli, run_id, headers=headers)
                poll = await cli.get(f"/v1/runs/{run_id}", headers=headers)
                status = await poll.json()

        assert assistant_attempts == 2
        assert [event["event"] for event in terminals] == ["run.failed"]
        assert status["status"] == "failed"
        assert status["last_event"] == "run.failed"
        assert "session" in status["error"].lower()
        durable = auth_adapter._session_db.get_messages_as_conversation(session_id)
        assert [message["content"] for message in durable].count("hello") == 1
        assert not any(
            message["content"] == "CODEX_API_ASSISTANT" for message in durable
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_mode", ["returned-error", "exception"])
    async def test_canonical_codex_turn_failures_emit_one_failed_without_output_row(
        self, auth_adapter, failure_mode
    ):
        session_id = auth_adapter._session_db.create_session(
            f"codex-{failure_mode}", "api_server"
        )
        agent = _make_production_codex_agent(auth_adapter._session_db, session_id)
        if failure_mode == "exception":
            agent._codex_session.run_turn.side_effect = RuntimeError("codex crashed")
        else:
            turn = agent._codex_session.run_turn.return_value
            turn.error = "codex timed out"
            turn.final_text = "FAILED_CODEX_OUTPUT"
            turn.projected_messages = [
                {"role": "assistant", "content": "FAILED_CODEX_OUTPUT"}
            ]
        app = _create_runs_app(auth_adapter)
        headers = {"Authorization": "Bearer sk-secret"}

        with patch.object(auth_adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": session_id},
                    headers=headers,
                )
                run_id = (await response.json())["run_id"]
                terminals = await _subscribe_to_run(cli, run_id, headers=headers)
                poll = await cli.get(f"/v1/runs/{run_id}", headers=headers)
                status = await poll.json()

        assert [event["event"] for event in terminals] == ["run.failed"]
        assert status["status"] == "failed"
        assert status["last_event"] == "run.failed"
        assert [
            (message["role"], message["content"])
            for message in auth_adapter._session_db.get_messages_as_conversation(
                session_id
            )
        ] == [("user", "hello")]

    @pytest.mark.asyncio
    async def test_stop_racing_codex_turn_error_reports_failed_not_cancelled(
        self, auth_adapter
    ):
        session_id = auth_adapter._session_db.create_session(
            "codex-error-stop-race", "api_server"
        )
        agent = _make_production_codex_agent(auth_adapter._session_db, session_id)
        turn = agent._codex_session.run_turn.return_value
        turn.error = "codex timed out"
        turn.final_text = "FAILED_CODEX_OUTPUT"
        turn.projected_messages = [
            {"role": "assistant", "content": "FAILED_CODEX_OUTPUT"}
        ]
        turn_started = threading.Event()
        allow_error = threading.Event()

        def _blocked_error(**_kwargs):
            turn_started.set()
            allow_error.wait(timeout=5)
            return turn

        agent._codex_session.run_turn.side_effect = _blocked_error
        app = _create_runs_app(auth_adapter)
        headers = {"Authorization": "Bearer sk-secret"}

        with patch.object(auth_adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": session_id},
                    headers=headers,
                )
                run_id = (await response.json())["run_id"]
                assert turn_started.wait(timeout=3)
                terminals_task = asyncio.create_task(
                    _subscribe_to_run(cli, run_id, headers=headers)
                )
                stop = await cli.post(f"/v1/runs/{run_id}/stop", headers=headers)
                allow_error.set()
                terminals = await terminals_task
                poll = await cli.get(f"/v1/runs/{run_id}", headers=headers)
                status = await poll.json()

        assert stop.status == 200
        assert [event["event"] for event in terminals] == ["run.failed"]
        assert status["status"] == "failed"
        assert status["last_event"] == "run.failed"
        assert [
            (message["role"], message["content"])
            for message in auth_adapter._session_db.get_messages_as_conversation(
                session_id
            )
        ] == [("user", "hello")]

    @pytest.mark.asyncio
    async def test_stop_during_failed_codex_flush_reports_failed_not_cancelled(
        self, auth_adapter
    ):
        session_id = auth_adapter._session_db.create_session(
            "codex-failed-stop-race", "api_server"
        )
        agent = _make_production_codex_agent(auth_adapter._session_db, session_id)
        real_flush = agent._flush_messages_to_session_db
        persistence_started = threading.Event()
        allow_failure = threading.Event()

        def _blocked_failed_flush(messages, conversation_history=None):
            if any(
                message.get("content") == "CODEX_API_ASSISTANT"
                for message in messages
                if isinstance(message, dict)
            ):
                persistence_started.set()
                allow_failure.wait(timeout=5)
                return False
            return real_flush(messages, conversation_history)

        agent._flush_messages_to_session_db = _blocked_failed_flush
        app = _create_runs_app(auth_adapter)
        headers = {"Authorization": "Bearer sk-secret"}

        with patch.object(auth_adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": session_id},
                    headers=headers,
                )
                run_id = (await response.json())["run_id"]
                assert persistence_started.wait(timeout=3)
                events_task = asyncio.create_task(
                    _subscribe_to_run(cli, run_id, headers=headers)
                )
                stop = await cli.post(
                    f"/v1/runs/{run_id}/stop", headers=headers
                )
                allow_failure.set()
                terminals = await events_task
                poll = await cli.get(f"/v1/runs/{run_id}", headers=headers)
                status = await poll.json()

        assert stop.status == 200
        assert [event["event"] for event in terminals] == ["run.failed"]
        assert status["status"] == "failed"
        assert status["last_event"] == "run.failed"

    @pytest.mark.asyncio
    async def test_completion_stop_race_emits_only_cancelled(self, adapter):
        app = _create_runs_app(adapter)
        started = threading.Event()
        finish = threading.Event()
        with patch.object(adapter, "_create_agent") as mock_create:
            agent = MagicMock()

            def _run(*_args, **_kwargs):
                started.set()
                finish.wait(timeout=5)
                return {
                    "final_response": "late success",
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "late success"},
                    ],
                }

            agent.run_conversation.side_effect = _run
            agent.interrupt.side_effect = lambda *_args: finish.set()
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            mock_create.return_value = agent

            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await response.json())["run_id"]
                assert started.wait(timeout=3)
                events_task = asyncio.create_task(_subscribe_to_run(cli, run_id))
                stop = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop.status == 200
                terminals = await events_task

        assert [event["event"] for event in terminals] == ["run.cancelled"]
        agent._persist_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_during_successful_persistence_reports_completed_everywhere(
        self, auth_adapter
    ):
        session_id = auth_adapter._session_db.create_session(
            "persist-race", "api_server"
        )
        app = _create_runs_app(auth_adapter)
        persistence_started = threading.Event()
        allow_persistence = threading.Event()

        with patch.object(auth_adapter, "_create_agent") as mock_create:
            agent = MagicMock()

            def _run(*_args, **_kwargs):
                messages = [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "committed answer"},
                ]
                persistence_started.set()
                allow_persistence.wait(timeout=5)
                auth_adapter._session_db.append_messages_batch(
                    session_id,
                    messages=messages,
                    turn_lease_holder=agent._active_session_turn_lease_holder,
                )
                agent._turn_persistence_callback(
                    succeeded=True, successful_turn=True
                )
                return {"final_response": "committed answer", "messages": messages}

            agent.run_conversation.side_effect = _run
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            mock_create.return_value = agent

            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": session_id},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert response.status == 202
                run_id = (await response.json())["run_id"]
                assert persistence_started.wait(timeout=3)
                events_task = asyncio.create_task(
                    _subscribe_to_run(
                        cli, run_id, headers={"Authorization": "Bearer sk-secret"}
                    )
                )
                stop = await cli.post(
                    f"/v1/runs/{run_id}/stop",
                    headers={"Authorization": "Bearer sk-secret"},
                )
                allow_persistence.set()
                terminals = await events_task
                poll = await cli.get(
                    f"/v1/runs/{run_id}",
                    headers={"Authorization": "Bearer sk-secret"},
                )
                status = await poll.json()

        assert stop.status == 200
        assert [event["event"] for event in terminals] == ["run.completed"]
        assert status["status"] == "completed"
        assert status["last_event"] == "run.completed"
        assert [
            (message["role"], message["content"])
            for message in auth_adapter._session_db.get_messages_as_conversation(
                session_id
            )
        ] == [("user", "hello"), ("assistant", "committed answer")]


    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        auth_adapter._session_db.create_session("shared-project", "api_server")
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "victim",
                        "session_id": "shared-project",
                        "conversation_history": [],
                    },
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "attacker",
                        "session_id": "shared-project",
                        "conversation_history": [],
                    },
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                victim_ready.wait(timeout=3.0)
                attacker_ready.wait(timeout=3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/steer — steer a running agent
# ---------------------------------------------------------------------------


class TestSteerRun:
    @pytest.mark.asyncio
    async def test_steer_running_agent(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        queue = asyncio.Queue()
        adapter._active_run_agents["run_123"] = agent
        adapter._run_streams["run_123"] = queue
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": "tighten the ending"})
            payload = await resp.json()

        assert resp.status == 200
        assert payload == {
            "object": "hermes.run.steer",
            "run_id": "run_123",
            "accepted": True,
        }
        agent.steer.assert_called_once_with("tighten the ending")
        assert adapter._run_statuses["run_123"]["last_event"] == "run.steered"
        event = queue.get_nowait()
        assert event["event"] == "run.steered"
        assert event["run_id"] == "run_123"
        assert event["accepted"] is True

    @pytest.mark.asyncio
    async def test_steer_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_missing/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 404
        assert payload["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_steer_inactive_run_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        adapter._set_run_status("run_done", "completed")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_done/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 409
        assert payload["error"]["code"] == "run_not_accepting_steer"

    @pytest.mark.asyncio
    async def test_steer_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_123"] = agent
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": ""})
            payload = await resp.json()

        assert resp.status == 400
        assert payload["error"]["code"] == "invalid_steer_input"
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_then_steer_rejects_retained_agent_ref(self, adapter):
        """Steer must reject a stopping run even if the executor thread is still live."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_started = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.steer = MagicMock(return_value=True)

                def _interrupt(_message=None):
                    return None

                def _run_conversation(*_args, **_kwargs):
                    run_started.set()
                    run_can_finish.wait(timeout=5)
                    return {"final_response": "late result"}

                mock_agent.interrupt = MagicMock(side_effect=_interrupt)
                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert run_started.wait(timeout=3.0)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                assert run_id in adapter._active_run_agents

                steer_resp = await cli.post(
                    f"/v1/runs/{run_id}/steer",
                    json={"input": "tighten the ending"},
                )
                steer_data = await steer_resp.json()

                assert steer_resp.status == 409
                assert steer_data["error"]["code"] == "run_not_accepting_steer"
                mock_agent.steer.assert_not_called()

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_pending_steer_preserved_on_run_completed(self, adapter):
        """A steer drained by the turn finalizer (accepted after the final
        response) must surface as pending_steer on the terminal run status
        instead of being silently dropped."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.run_conversation.return_value = {
                    "final_response": "done",
                    "pending_steer": "tighten the ending",
                }
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]

                for _ in range(40):
                    status = adapter._run_statuses.get(run_id, {})
                    if status.get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert adapter._run_statuses[run_id]["status"] == "completed"
        assert adapter._run_statuses[run_id]["pending_steer"] == "tighten the ending"

    @pytest.mark.asyncio
    async def test_steer_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/steer", json={"input": "hello"})

        assert resp.status == 401


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:

    @pytest.mark.asyncio
    async def test_expired_live_run_drops_transport_but_keeps_control_state(self, adapter):
        """Stream TTL bounds buffering without detaching a live run."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id not in adapter._run_streams
                assert run_id not in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.2)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks


    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


class TestRunsProviderAuthFailure:
    @pytest.mark.asyncio
    async def test_status_reports_provider_auth_failure_distinctly(self, adapter):
        """/v1/runs builds its own agent via _create_agent() and does not
        route through _run_agent(), so the controlled "Provider
        authentication failed" message added there does not cover this
        endpoint. _handle_runs()'s own _ProviderAuthResolutionError branch
        must give the same distinguished message instead of the generic
        except-Exception "run failed" text."""
        from gateway.platforms.api_server import _ProviderAuthResolutionError

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.side_effect = _ProviderAuthResolutionError(
                    "No credentials found for provider 'nous'"
                )

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "failed"
                assert status["error"] == "⚠️ Provider authentication failed: No credentials found for provider 'nous'"
                assert status["last_event"] == "run.failed"
