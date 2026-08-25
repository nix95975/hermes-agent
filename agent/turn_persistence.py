"""Shared end-of-turn transcript persistence contract.

Both the standard conversation finalizer and early-return runtimes must use this
seam so retry, durable-turn arbitration, and callback reporting cannot drift.
"""

from __future__ import annotations

import logging
from typing import Any, Optional


def persist_turn_transcript(
    agent: Any,
    messages: list[dict],
    conversation_history: Optional[list[dict]] = None,
    *,
    successful_turn: bool,
    logger: logging.Logger,
    log_context: str,
) -> Optional[Exception]:
    """Persist one projected transcript, retrying under the caller's turn lease.

    ``agent._persist_session`` owns marker-aware projection to the durable DB.
    Calling it again is therefore safe and avoids a second, competing transcript
    projection.  The caller must keep its durable session-turn lease active for
    this function's full lifetime.

    Returns the final exception when every attempt fails, otherwise ``None``.
    The per-turn callback is reported exactly once for both outcomes.  Disabled
    or unavailable persistence remains a successful no-op because
    ``_persist_session`` preserves that established contract.
    """
    attempts = max(
        1, int(getattr(agent, "_turn_persist_retry_attempts", 1) or 1)
    )
    persist_error: Optional[Exception] = None
    succeeded = False

    for attempt in range(attempts):
        try:
            agent._persist_session(messages, conversation_history)
            persist_error = None
            succeeded = True
            break
        except Exception as exc:
            persist_error = exc
            if attempt + 1 < attempts:
                logger.warning(
                    "%s: _persist_session failed; retrying under the active "
                    "turn lease: %s",
                    log_context,
                    exc,
                )

    callback = getattr(agent, "_turn_persistence_callback", None)
    if callable(callback):
        try:
            callback(succeeded=succeeded, successful_turn=successful_turn)
        except Exception:
            logger.debug("turn persistence callback failed", exc_info=True)

    return persist_error
