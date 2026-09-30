"""Quiet ``hermes chat -Q`` helpers: bind this session's key and resume nested notifies.

Bot Mode delivers a local DM as ``hermes -p <bot> chat -Q --query-file``. Interactive
chat binds ``set_current_session_key(self.session_id)`` around the turn; the quiet
path did not, so a nested ``message_agent`` notify inherited the dispatcher's
``HERMES_SESSION_KEY`` and never woke the recipient. Quiet also printed and exited
after one turn, so a nested teammate reply that finished during the one-shot linger
was never injected as a follow-up.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, MutableMapping

# Nested A→B→C is one extra turn; this caps a runaway message_agent chain.
_MAX_QUIET_NOTIFY_ROUNDS = 8
QUIET_BOT_DELIVERY_POLL_SECONDS = 0.5
QUIET_BOT_DELIVERY_INTERRUPTED = object()

# Last line a Kanban worker leaves in its own log: ``[kanban-worker-exit] rc=<code>``. A per-tick
# ``hermes kanban dispatch`` process never reaped the worker, so ``os.waitpid`` cannot tell it how
# the worker exited; the trailer is the process-independent witness the dead-worker sweep reads
# instead, so a clean exit without a terminal board call is booked as the same protocol violation
# (and a 75 as the same rate-limit requeue) whichever process notices the death.
KANBAN_WORKER_EXIT_TRAILER = "[kanban-worker-exit] rc="


def exit_single_query(code: int) -> None:
    """``sys.exit(code)`` for a one-shot turn; a Kanban worker first writes the exit trailer to its log."""
    if os.environ.get("HERMES_KANBAN_TASK"):
        with contextlib.suppress(Exception):
            # stderr: stdout may be the ``--stream-json`` record stream, and the worker log
            # captures both streams.
            print(f"\n{KANBAN_WORKER_EXIT_TRAILER}{int(code)}", file=sys.stderr, flush=True)
    sys.exit(code)


# A spawner that bounds only the TURN (the cron Bot Chat lane) hands the quiet child a report
# path here. The child records the turn's outcome there the moment the turn ends, BEFORE the
# one-shot exit linger, so the spawner can book the delivery and stop waiting while the linger
# keeps protecting nested ``notify_on_complete`` replies. Popped before the turn runs (same
# contract as HERMES_TURN_AUTHOR): nothing the turn spawns inherits it, and a nested one-shot
# never writes over its host's report — the record also carries the writer's pid.
TURN_REPORT_FILE_ENV = "HERMES_QUIET_TURN_REPORT_FILE"


def take_turn_report_path(environ: MutableMapping[str, str] = os.environ) -> str | None:
    """Read and remove the spawner's turn-report path so subprocesses started during the turn do not inherit it."""
    return environ.pop(TURN_REPORT_FILE_ENV, None) or None


def write_turn_report(path: str | None, *, exit_code: int, error: str = "", reply: str = "") -> None:
    """Atomically record ``{pid, exit_code, error, reply}`` at *path*; a no-op without a path. Never
    raises: the report is the spawner's convenience, the turn itself is already persisted. ``reply``
    is what the run will print — a spawner booking a lingering child from its report relays it."""
    if not path:
        return
    from utils import atomic_json_write

    record = {"pid": os.getpid(), "exit_code": int(exit_code), "error": str(error or ""), "reply": str(reply or "")}
    # 0600 from creation: the record now carries the turn's answer, like the 0600 query file beside it.
    with contextlib.suppress(Exception):
        atomic_json_write(path, record, indent=None, mode=0o600)


def read_turn_report(path: str, pid: int) -> dict | None:
    """The child's turn report, or None while absent, unreadable, or written by another process."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("pid") != pid:
        return None
    return record


# After the child reports its turn, a child with nothing to linger for exits at once; a spawner
# that needs only the outcome gives it that long so its real exit code and stream tails are
# booked instead of the report's summary.
REPORTED_TURN_EXIT_GRACE_SECONDS = 2.0


def run_reported_turn(argv: list, *, env: MutableMapping[str, str], report_path: str, timeout: float,
                      exit_grace: float | None = REPORTED_TURN_EXIT_GRACE_SECONDS, cwd: str | None = None,
                      encoding: str | None = None) -> subprocess.CompletedProcess:
    """Run one ``hermes chat -Q`` delivery child; *timeout* bounds the TURN, not the process.

    The child records its turn at *report_path* (``write_turn_report``) the moment the turn ends,
    then runs the one-shot exit linger for nested ``notify_on_complete`` replies — bounded by
    ``terminal.oneshot_completion_wait_seconds``, whose default equals the delivery caps, so
    waiting for process exit booked every delivered turn that left a reply pending as a timeout
    and killed the linger (#113608, #114980). A child that exits is booked from its real exit
    code and streams. A child still lingering once its report exists is booked from the report
    and left running (a daemon thread drains and reaps it): after *exit_grace* seconds for a
    spawner that needs only the outcome, or at the cap when *exit_grace* is None, for a spawner
    that relays the printed answer — a teammate's reply during the linger may still become it.
    Only a turn that never ends is killed, as ``subprocess.TimeoutExpired``.

    *cwd* pins the child's directory (a spawner sitting in a reaped scratch workspace must not
    hand its dead cwd on — the child dies at CLI startup, #102941). The pipes decode lossily
    everywhere: a stray non-UTF-8 byte (a grandchild sharing the pipe interleaving a partial
    multi-byte write) must not raise in the drain thread and take the reply and the failure tail
    with it (#105582). Without an explicit *encoding* they decode as UTF-8 only on win32, where
    the child is guaranteed UTF-8 (hermes_bootstrap reconfigures its streams even under
    PYTHONIOENCODING=cp1252) while the gateway parent is not started in UTF-8 mode, so the
    locale default mangled or lost accented replies (#115894); on POSIX the child keeps the
    locale codec, so the locale default stays correct there (#66566).
    """
    from hermes_cli._subprocess_compat import windows_hide_flags

    if encoding is None and sys.platform == "win32":
        encoding = "utf-8"
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding=encoding, errors="replace", env={**env, TURN_REPORT_FILE_ENV: report_path},
        cwd=cwd, creationflags=windows_hide_flags())
    streams: dict = {}

    def _drain() -> None:
        streams["out"], streams["err"] = proc.communicate()

    drain = threading.Thread(target=_drain, name=f"quiet-turn-drain-{proc.pid}", daemon=True)
    drain.start()
    deadline = time.monotonic() + timeout
    report = None
    while True:
        drain.join(timeout=exit_grace if report is not None and exit_grace is not None else 0.25)
        if not drain.is_alive():
            return subprocess.CompletedProcess(argv, proc.returncode, streams.get("out", ""), streams.get("err", ""))
        if report is not None and exit_grace is not None:
            break
        # Re-read while waiting for the cap: a follow-up turn rewrites the report with its answer.
        report = read_turn_report(report_path, proc.pid) or report
        if time.monotonic() >= deadline:
            if report is not None:
                break
            proc.kill()
            drain.join(timeout=5.0)
            # A killed child cannot run further, but the turn may have ENDED (and delivered)
            # in the window between the last report check and the kill landing. Re-read once:
            # a report that appeared means the turn completed — book it instead of
            # misreporting a delivered turn as a timeout (and never re-notifying).
            report = read_turn_report(report_path, proc.pid)
            if report is not None:
                break
            raise subprocess.TimeoutExpired(argv, timeout)
    # Turn over, child still lingering for a nested reply: not this spawner's wait.
    return subprocess.CompletedProcess(
        argv, int(report["exit_code"]), report.get("reply") or "", report.get("error") or "")


@contextlib.contextmanager
def bind_quiet_session_key(session_id: str):
    """Bind the approval/session key to *this* quiet session for the enclosing ``with`` block."""
    from tools.approval_context import reset_current_session_key, set_current_session_key

    token = set_current_session_key(session_id or "default")
    try:
        yield
    finally:
        reset_current_session_key(token)


def _diagnostic_only_wake_muted(events) -> bool:
    """True when every drained event is an automatic diagnostic AND the CLI policy suppresses them."""
    from agent.notification_presentation import diagnostic_process_event
    from gateway.warning_notifications import warning_notifications_enabled

    if not events or not all(diagnostic_process_event(e) for e in events if isinstance(e, dict)):
        return False
    return not warning_notifications_enabled("cli")


def quiet_notify_linger_seconds() -> float:
    """Total linger budget for one quiet run: the shared ``terminal.oneshot_completion_wait_seconds``.

    One budget covers the drain loop here AND the later ``_wait_for_oneshot_background_completions``
    pass, so a stuck ``notify_on_complete`` child cannot stack round-after-round waits on top of the
    finalize re-wait (pre-fix worst case: 8 rounds x 600s + 600s).
    """
    from tools.process_registry import ProcessRegistry

    return ProcessRegistry._oneshot_completion_wait_seconds()


def start_quiet_cli_live_delivery(cli: Any, *, turn_author_supported: bool) -> bool:
    """Advertise only this quiet CLI poller when it owns the profile's canonical Bot Chat."""
    lease = getattr(cli, "_active_session_lease", None)
    session_id = str(getattr(cli, "session_id", "") or "")
    agent_db = getattr(getattr(cli, "agent", None), "_session_db", None)
    db_path = getattr(agent_db, "db_path", None) or getattr(getattr(cli, "_session_db", None), "db_path", None)
    if not turn_author_supported or lease is None or not session_id or not db_path:
        return False
    from pathlib import Path
    from tools.bot_live_delivery import advertise_cli_consumer

    home = Path(db_path).resolve().parent
    live_session_id = str(getattr(cli, "_quiet_bot_live_session_id", "") or session_id)
    owner = advertise_cli_consumer(home, lease, session_id=session_id, live_session_id=live_session_id)
    if owner is None:
        return False
    cli._quiet_bot_live_delivery = {"home": home, "live_session_id": live_session_id}
    cli._quiet_bot_live_session_id = live_session_id
    cli._quiet_bot_live_delivery_lost = False
    return True


def sync_quiet_cli_live_delivery(cli: Any) -> bool:
    state = getattr(cli, "_quiet_bot_live_delivery", None)
    if not state:
        return False
    from tools.bot_live_delivery import sync_cli_consumer

    synced = sync_cli_consumer(
        state["home"], getattr(cli, "_active_session_lease", None),
        session_id=str(getattr(cli, "session_id", "") or ""),
        live_session_id=state["live_session_id"],
    )
    if not synced:
        cli._quiet_bot_live_delivery_lost = True
        abort_quiet_cli_live_delivery(cli, reason="CLI no longer owns the canonical Bot Chat tip")
    return synced


def poll_quiet_cli_live_delivery_once(cli: Any, history: list, *, turn_author_supported: bool = True):
    """Run one claimed mailbox turn at an idle boundary and return its updated shared history."""
    state = getattr(cli, "_quiet_bot_live_delivery", None)
    if not state:
        return None
    from tools.bot_live_delivery import (
        claim_pending_delivery, complete_delivery, find_canonical_live_owner,
    )

    if (getattr(cli, "_quiet_bot_live_delivery_lost", False)
            or not sync_quiet_cli_live_delivery(cli)):
        return None
    home = state["home"]
    owner = find_canonical_live_owner(home, include_cli=True)
    lease = getattr(cli, "_active_session_lease", None)
    if (owner is None or lease is None or owner.get("lease_id") != lease.lease_id
            or owner.get("live_session_id") != state["live_session_id"]):
        return None
    claimed = claim_pending_delivery(home, owner)
    if claimed is None:
        return None

    delivery_id = str(claimed["delivery_id"])
    author = claimed.get("author") or None
    kwargs = {}
    if author and turn_author_supported:
        kwargs["turn_author"] = author
    try:
        result = cli.agent.run_conversation(
            user_message=claimed["message"], conversation_history=history, **kwargs,
        )
    except KeyboardInterrupt:
        complete_delivery(home, delivery_id, status="cancelled", error="interrupted", reason="cancelled")
        raise
    except Exception as exc:
        error = str(exc)
        from tools.bot_failure_reasons import classify_agent_error

        complete_delivery(home, delivery_id, status="failed", error=error,
                          reason=classify_agent_error(error))
        return history

    updated_history = result.get("messages") if isinstance(result, dict) else None
    if isinstance(updated_history, list) and updated_history:
        history = updated_history
        cli.conversation_history = history
    if isinstance(result, dict) and result.get("interrupted"):
        complete_delivery(home, delivery_id, status="cancelled", error="interrupted", reason="cancelled")
        from cli import _sync_cli_session_id_from_agent

        _sync_cli_session_id_from_agent(cli)
        abort_quiet_cli_live_delivery(cli, reason="mailbox turn interrupted")
        return QUIET_BOT_DELIVERY_INTERRUPTED
    failed = (not isinstance(result, dict) or result.get("failed") or result.get("partial")
              or result.get("completed") is False)
    if failed:
        error = str(result.get("error") or "mailbox turn did not complete") if isinstance(result, dict) else "mailbox turn did not run"
        from tools.bot_failure_reasons import classify_agent_error

        complete_delivery(home, delivery_id, status="failed", error=error,
                          reason=classify_agent_error(error))
    else:
        reply = str(result.get("final_response") or "")
        from gateway.response_filters import is_intentional_silence_response

        if is_intentional_silence_response(reply):
            reply = ""
        complete_delivery(home, delivery_id, status="settled", reply=reply)
    from cli import _sync_cli_session_id_from_agent

    _sync_cli_session_id_from_agent(cli)
    return history


def abort_quiet_cli_live_delivery(cli: Any, *, reason: str) -> None:
    """Withdraw an interrupted one-shot consumer without starting another model turn."""
    state = getattr(cli, "_quiet_bot_live_delivery", None)
    if not state:
        return
    from tools.bot_live_delivery import abort_cli_consumer

    abort_cli_consumer(
        state["home"], getattr(cli, "_active_session_lease", None),
        live_session_id=state["live_session_id"], reason=reason,
    )
    cli._quiet_bot_live_delivery = None


def continue_quiet_notify_completions(
    session_id: str,
    run_turn: Callable[[str], Any],
    *,
    owns_event=None,
    max_rounds: int = _MAX_QUIET_NOTIFY_ROUNDS,
    linger_budget: float | None = None,
    on_idle: Callable[[], Any] | None = None,
    on_finish: Callable[[bool], Any] | None = None,
    idle_poll_seconds: float = QUIET_BOT_DELIVERY_POLL_SECONDS,
) -> Any:
    """Linger for ``notify_on_complete`` work, then run owned completion texts as follow-up turns.

    Returns the last ``run_turn`` result, or ``None`` when nothing owned completed. The whole
    loop shares ONE linger budget (default: ``terminal.oneshot_completion_wait_seconds``) — a
    process that times out is waited on no further this run: after the current round's drained
    texts run, the loop stops (the finalize linger still covers it once, bounded, via the
    budget handshake below).
    """
    from tools.process_registry import process_registry
    from tools.async_delegation import claim_event_delivery, complete_event_delivery

    last: Any = None
    finish_called = False
    key = session_id or ""
    if linger_budget is None:
        linger_budget = quiet_notify_linger_seconds()
    deadline = time.monotonic() + max(float(linger_budget), 0.0)
    for round_index in range(max_rounds):
        while True:
            remaining = max(deadline - time.monotonic(), 0.0)
            if remaining <= 0:
                wait = (process_registry.wait_for_pending_completions(None, timeout=0.0)
                        if on_idle is None else {"timed_out": True})
                break
            wait_budget = min(remaining, max(float(idle_poll_seconds), 0.01)) if on_idle else remaining
            wait = process_registry.wait_for_pending_completions(None, timeout=wait_budget)
            idle_result = on_idle() if on_idle is not None else None
            if idle_result is QUIET_BOT_DELIVERY_INTERRUPTED:
                wait = {"interrupted": True}
                break
            if idle_result is True:
                # A mailbox turn may have started notify_on_complete work. Re-enter the
                # same wait loop before draining follow-ups or attempting the final close.
                continue
            if wait.get("interrupted"):
                break
            # A short poll timeout is only a chance to check the mailbox; preserve the
            # original absolute linger deadline and the existing notify round budget.
            if not (on_idle is not None and wait.get("timed_out")
                    and time.monotonic() < deadline):
                break
        if wait.get("interrupted"):
            if on_finish is not None:
                on_finish(False)
                finish_called = True
            break
        drained = []
        for event, text in process_registry.drain_notifications(session_key=key, owns_event=owns_event):
            # Durable async_delegation events carry a delivery ledger: without the
            # claim/complete handshake the row stays delivery_state='pending' and
            # restore_undelivered_completions re-queues it on the next process start,
            # injecting the same result twice. Same contract as every other drain consumer.
            claim = claim_event_delivery(event, "cli-quiet")
            if claim is None:
                continue
            complete_event_delivery(event, claim)
            drained.append((event, text))
        # Every drained event type carries formatted text (completions, watch matches,
        # async_delegation results): drain_notifications POPS owned events off the queue,
        # so filtering by type here would consume-and-silently-drop owned
        # async_delegation results. Keep everything that rendered.
        texts = [text for _event, text in drained if text]
        if texts:
            follow = run_turn("\n\n".join(texts))
            # Same admission rule as the interactive CLI turn: a wake made ONLY of automatic
            # diagnostics (early failure / watch notices) still runs, but under suppression its
            # reply never displaces the requested one-shot answer on stdout.
            if not _diagnostic_only_wake_muted([event for event, text in drained if text]):
                last = follow
        if wait.get("timed_out"):
            if on_finish is not None:
                on_finish(False)
                finish_called = True
            break
        if not texts:
            can_continue = round_index + 1 < max_rounds and time.monotonic() < deadline
            if on_finish is not None and on_finish(can_continue) is True:
                continue
            finish_called = on_finish is not None
            return last
    if on_finish is not None and not finish_called:
        on_finish(False)
    return last


def adopt_unanswered_turn(cli: Any, query: Any, environ: MutableMapping[str, str] = os.environ) -> bool:
    """A dispatcher's re-run of a failed delivery turn resumes the DM its first attempt already
    persisted instead of appending it again. Returns True when the tail row was adopted.

    The failed attempt's turn-start persist left the DM as the transcript's unanswered tail row. A
    fresh process cannot know that by itself (``_DB_PERSISTED_MARKER`` is in-process only), and
    inferring it from an identical tail alone would swallow a person's deliberate re-send — so the
    dispatcher must say so with ``tools.bot_relay.RESUME_UNANSWERED_TURN_ENV``, consumed (popped) here
    before the turn so tool subprocesses never inherit it. Which row counts as the unanswered DM, and
    how it is re-staged as ``_pending_cli_user_message``, is shared with the in-process peer-DM lane
    (``agent.session_persistence.adopt_unanswered_turn``, #115325).
    """
    from tools.bot_relay import RESUME_UNANSWERED_TURN_ENV

    if environ.pop(RESUME_UNANSWERED_TURN_ENV, None) != "1":
        return False
    from agent.session_persistence import adopt_unanswered_turn as _adopt_tail

    return _adopt_tail(getattr(cli, "conversation_history", None) or [], query, cli.agent)
