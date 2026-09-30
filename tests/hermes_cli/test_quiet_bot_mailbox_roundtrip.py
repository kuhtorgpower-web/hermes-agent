"""Regression for a quiet Bot Chat mailbox cycle across two profile homes."""

from __future__ import annotations

import io
import shlex
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
import hermes_constants
from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB
from tools import bot_live_delivery as mailbox
from tools import process_registry as process_registry_module
from tools.process_registry import process_registry


def test_quiet_cli_mailbox_roundtrip_across_profiles_while_both_wait(
        tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    assert Path(cli.__file__).resolve().is_relative_to(repo_root), cli.__file__
    assert Path(mailbox.__file__).resolve().is_relative_to(repo_root), mailbox.__file__
    assert Path(process_registry_module.__file__).resolve().is_relative_to(repo_root)

    home_a = tmp_path / "profiles" / "alpha"
    home_b = tmp_path / "profiles" / "beta"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)

    db_a = SessionDB(db_path=home_a / "state.db")
    db_b = SessionDB(db_path=home_b / "state.db")
    db_a.create_session(session_id="chat-alpha", source="cli")
    db_a.set_session_title("chat-alpha", "Bot Chat")
    db_b.create_session(session_id="chat-beta", source="cli")
    db_b.set_session_title("chat-beta", "Bot Chat")
    lease_a, refusal_a = try_acquire_active_session(
        session_id="chat-alpha", surface="cli", config={}, registry_home=home_a,
        metadata={"live_session_id": "live-alpha"},
    )
    lease_b, refusal_b = try_acquire_active_session(
        session_id="chat-beta", surface="cli", config={}, registry_home=home_b,
        metadata={"live_session_id": "live-beta"},
    )
    assert refusal_a is None and lease_a is not None
    assert refusal_b is None and lease_b is not None

    author_a = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
    author_b = {"id": "bot:beta", "name": "Beta", "is_bot": True}
    delivery_ids: dict[str, str] = {}
    calls_a: list[dict] = []
    calls_b: list[dict] = []
    failures: dict[str, BaseException] = {}
    exit_codes: dict[str, int] = {}

    beta_advertised = threading.Event()
    alpha_to_beta_admitted = threading.Event()
    beta_to_alpha_admitted = threading.Event()
    alpha_to_beta_settled = threading.Event()

    class ThreadWriter:
        """Route concurrent CLI prints to the output buffer owned by that thread."""

        def __init__(self, fallback):
            self.fallback = fallback
            self.local = threading.local()

        @contextmanager
        def capture(self, stream):
            previous = getattr(self.local, "stream", None)
            self.local.stream = stream
            try:
                yield
            finally:
                if previous is None:
                    del self.local.stream
                else:
                    self.local.stream = previous

        def write(self, text):
            stream = getattr(self.local, "stream", None) or self.fallback
            return stream.write(text)

        def flush(self):
            stream = getattr(self.local, "stream", None) or self.fallback
            return stream.flush()

        def __getattr__(self, name):
            return getattr(self.fallback, name)

    stdout_a, stdout_b = io.StringIO(), io.StringIO()
    stderr_a, stderr_b = io.StringIO(), io.StringIO()
    stdout_writer = ThreadWriter(sys.stdout)
    stderr_writer = ThreadWriter(sys.stderr)
    monkeypatch.setattr(sys, "stdout", stdout_writer)
    monkeypatch.setattr(sys, "stderr", stderr_writer)

    def result(message, answer, history):
        return {
            "final_response": answer,
            "messages": list(history) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ],
        }

    def run_alpha_conversation(*, user_message, conversation_history, **kwargs):
        calls_a.append({
            "message": user_message,
            "home": hermes_constants.get_hermes_home().resolve(),
            "history": list(conversation_history),
            "author": kwargs.get("turn_author"),
        })
        if user_message == "alpha original question":
            assert calls_a[-1]["home"] == home_a.resolve()
            owner_b = mailbox.find_canonical_live_owner(home_b, include_cli=True)
            assert owner_b is not None, "profile B quiet CLI did not advertise its busy consumer"
            assert owner_b["profile_home"] == str(home_b.resolve())
            assert owner_b["session_id"] == "chat-beta"
            assert owner_b["lease_id"] == lease_b.lease_id
            delivery = mailbox.deliver_to_live_owner(
                home_b, owner_b, "question from alpha", author=author_a,
            )
            delivery_ids["alpha_to_beta"] = delivery["delivery_id"]
            assert mailbox.read_delivery_result(home_b, delivery["delivery_id"])["status"] == "queued"
            assert mailbox.read_delivery_result(home_a, delivery["delivery_id"]) is None
            alpha_to_beta_admitted.set()
            return result(user_message, "alpha original answer", conversation_history)

        assert user_message == "reply from beta"
        assert calls_a[-1]["home"] == home_a.resolve(), "the reply turn ran in the sender profile"
        assert calls_a[-1]["author"] == author_b
        ticket = mailbox.read_delivery_result(home_a, delivery_ids["beta_to_alpha"])
        assert ticket["status"] == "claimed", "profile A must claim the real reverse ticket before execution"
        assert any(row.get("content") == "alpha original answer" for row in calls_a[-1]["history"])
        return result(user_message, "alpha handled beta reply", conversation_history)

    def run_beta_conversation(*, user_message, conversation_history, **kwargs):
        calls_b.append({
            "message": user_message,
            "home": hermes_constants.get_hermes_home().resolve(),
            "history": list(conversation_history),
            "author": kwargs.get("turn_author"),
        })
        if user_message == "beta original question":
            assert calls_b[-1]["home"] == home_b.resolve()
            owner_b = mailbox.find_canonical_live_owner(home_b, include_cli=True)
            assert owner_b is not None and owner_b["lease_id"] == lease_b.lease_id
            beta_advertised.set()
            return result(user_message, "beta original answer", conversation_history)

        assert user_message == "question from alpha"
        assert calls_b[-1]["home"] == home_b.resolve(), "profile A's ticket ran under the wrong home"
        assert calls_b[-1]["author"] == author_a
        ticket = mailbox.read_delivery_result(home_b, delivery_ids["alpha_to_beta"])
        assert ticket["status"] == "claimed", "profile B must claim the real inbound ticket before execution"
        assert any(row.get("content") == "beta original answer" for row in calls_b[-1]["history"])

        owner_a = mailbox.find_canonical_live_owner(home_a, include_cli=True)
        assert owner_a is not None and owner_a["lease_id"] == lease_a.lease_id
        reverse = mailbox.deliver_to_live_owner(
            home_a, owner_a, "reply from beta", author=author_b,
        )
        delivery_ids["beta_to_alpha"] = reverse["delivery_id"]
        assert mailbox.read_delivery_result(home_a, reverse["delivery_id"])["status"] == "queued"
        assert mailbox.read_delivery_result(home_b, reverse["delivery_id"]) is None
        beta_to_alpha_admitted.set()

        # Beta's mailbox turn remains active until Alpha's quiet idle poll settles this
        # second, real ticket. That creates the A→B→A wait cycle the CLI must break.
        reverse_receipt = mailbox.await_delivery(home_a, reverse["delivery_id"], timeout=3.0)
        assert reverse_receipt is not None and reverse_receipt["status"] == "settled"
        assert reverse_receipt["reply"] == "alpha handled beta reply"
        return result(user_message, "beta handled alpha request", conversation_history)

    def wait_for_pending_completions(_task_id=None, *, timeout=None, **_kwargs):
        from tools.approval_context import get_current_session_key

        current_session = get_current_session_key()
        current_home = hermes_constants.get_hermes_home().resolve()
        wait_seconds = min(max(float(timeout or 0.0), 0.0), 0.05)

        def _waited(*, timed_out):
            return {
                "waited": ["pending-notify"],
                "completed": [],
                "timed_out": ["pending-notify"] if timed_out else [],
            }

        if current_session == "chat-beta":
            assert current_home == home_b.resolve()
            delivery_id = delivery_ids.get("alpha_to_beta")
            if delivery_id:
                ticket = mailbox.read_delivery_result(home_b, delivery_id)
                if ticket and ticket["status"] == "settled":
                    alpha_to_beta_settled.set()
                    return _waited(timed_out=False)
            woke = alpha_to_beta_admitted.wait(wait_seconds)
            return _waited(timed_out=not woke)

        assert current_session == "chat-alpha"
        assert current_home == home_a.resolve()
        if alpha_to_beta_settled.is_set():
            return _waited(timed_out=False)
        reverse_id = delivery_ids.get("beta_to_alpha")
        if reverse_id:
            reverse = mailbox.read_delivery_result(home_a, reverse_id)
            if reverse and reverse["status"] == "settled":
                woke = alpha_to_beta_settled.wait(wait_seconds)
                return _waited(timed_out=not woke)
        woke = beta_to_alpha_admitted.wait(wait_seconds)
        return _waited(timed_out=not woke)

    monkeypatch.setattr(
        "hermes_cli.quiet_single_query.quiet_notify_linger_seconds", lambda: 4.0,
    )
    monkeypatch.setattr(
        "hermes_cli.quiet_single_query.QUIET_BOT_DELIVERY_POLL_SECONDS", 0.05,
    )
    monkeypatch.setattr(mailbox, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        process_registry_module.process_registry,
        "wait_for_pending_completions", wait_for_pending_completions,
    )

    alpha_agent = SimpleNamespace(
        run_conversation=run_alpha_conversation, session_id="chat-alpha", _session_db=db_a,
    )
    beta_agent = SimpleNamespace(
        run_conversation=run_beta_conversation, session_id="chat-beta", _session_db=db_b,
    )
    alpha_cli = SimpleNamespace(
        agent=alpha_agent, conversation_history=[], session_id="chat-alpha",
        _active_session_lease=lease_a,
    )
    beta_cli = SimpleNamespace(
        agent=beta_agent, conversation_history=[], session_id="chat-beta",
        _active_session_lease=lease_b,
    )

    def run_quiet(home, fake_cli, name, stdout, stderr):
        token = hermes_constants.set_hermes_home_override(home)
        try:
            with stdout_writer.capture(stdout), stderr_writer.capture(stderr):
                try:
                    cli._run_quiet_single_query(fake_cli, f"{name} original question")
                except SystemExit as exc:
                    exit_codes[name] = exc.code
                except BaseException as exc:
                    failures[name] = exc
        finally:
            hermes_constants.reset_hermes_home_override(token)

    beta_thread = threading.Thread(
        target=run_quiet, args=(home_b, beta_cli, "beta", stdout_b, stderr_b),
        name="quiet-beta", daemon=True,
    )
    beta_thread.start()
    try:
        assert beta_advertised.wait(3.0), "profile B quiet CLI did not reach its original turn"
        run_quiet(home_a, alpha_cli, "alpha", stdout_a, stderr_a)
        beta_thread.join(timeout=5.0)
        assert not beta_thread.is_alive(), "profile B did not finish after profile A settled its reply ticket"
    finally:
        if beta_thread.is_alive():
            beta_thread.join(timeout=5.0)

    try:
        assert not failures, {name: repr(exc) for name, exc in failures.items()}
        assert exit_codes == {"alpha": 0, "beta": 0}
        assert [call["message"] for call in calls_a] == [
            "alpha original question", "reply from beta",
        ]
        assert [call["message"] for call in calls_b] == [
            "beta original question", "question from alpha",
        ]
        assert calls_a[1]["author"] == author_b
        assert calls_b[1]["author"] == author_a
        assert calls_a[1]["home"] == home_a.resolve()
        assert calls_b[1]["home"] == home_b.resolve()

        receipt_ab = mailbox.read_delivery_result(home_b, delivery_ids["alpha_to_beta"])
        receipt_ba = mailbox.read_delivery_result(home_a, delivery_ids["beta_to_alpha"])
        assert receipt_ab["status"] == "settled"
        assert receipt_ab["reply"] == "beta handled alpha request"
        assert receipt_ba["status"] == "settled"
        assert receipt_ba["reply"] == "alpha handled beta reply"

        assert stdout_a.getvalue().strip() == "alpha original answer"
        assert stdout_b.getvalue().strip() == "beta original answer"
        assert "handled beta reply" not in stdout_a.getvalue()
        assert "handled alpha request" not in stdout_b.getvalue()
        assert mailbox.find_canonical_live_owner(home_a, include_cli=True) is None
        assert mailbox.find_canonical_live_owner(home_b, include_cli=True) is None
        assert lease_a.released is False and lease_b.released is False
    finally:
        lease_a.release()
        lease_b.release()
        db_a.close()
        db_b.close()


def test_final_close_mailbox_turn_waits_for_its_real_notify_child(
        tmp_path, monkeypatch, capsys):
    repo_root = Path(__file__).resolve().parents[2]
    assert Path(cli.__file__).resolve().is_relative_to(repo_root), cli.__file__
    assert Path(mailbox.__file__).resolve().is_relative_to(repo_root), mailbox.__file__
    assert Path(process_registry_module.__file__).resolve().is_relative_to(repo_root)

    home_a = tmp_path / "profiles" / "alpha"
    home_b = tmp_path / "profiles" / "beta"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)
    db = SessionDB(db_path=home_b / "state.db")
    db.create_session(session_id="chat-beta", source="cli")
    db.set_session_title("chat-beta", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat-beta", surface="cli", config={}, registry_home=home_b,
        metadata={"live_session_id": "live-beta"},
    )
    assert refusal is None and lease is not None

    author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
    delivery = []
    children = []
    calls = []
    marker = tmp_path / "background-child.done"
    linger_seconds = 3.0
    linger_started = []
    wait_records = []

    def linger_budget():
        linger_started.append(time.monotonic())
        return linger_seconds

    real_wait = process_registry.wait_for_pending_completions

    def observe_real_wait(task_id=None, *args, **kwargs):
        started = time.monotonic()
        result = real_wait(task_id, *args, **kwargs)
        wait_records.append({
            "started": started,
            "timeout": kwargs.get("timeout"),
            "elapsed": time.monotonic() - started,
            "result": result,
        })
        return result

    real_close = mailbox.close_cli_consumer

    def admit_at_final_close(profile_home, active_lease, *, live_session_id):
        if not delivery:
            owner = mailbox.find_canonical_live_owner(home_b, include_cli=True)
            assert owner is not None and owner["lease_id"] == lease.lease_id
            token = hermes_constants.set_hermes_home_override(home_a)
            try:
                assert hermes_constants.get_hermes_home().resolve() == home_a.resolve()
                record = mailbox.deliver_to_live_owner(
                    home_b, owner, "message admitted at final close", author=author,
                )
                delivery.append(record)
                assert mailbox.read_delivery_result(home_b, record["delivery_id"])["status"] == "queued"
            finally:
                hermes_constants.reset_hermes_home_override(token)
        return real_close(profile_home, active_lease, live_session_id=live_session_id)

    monkeypatch.setattr("hermes_cli.quiet_single_query.quiet_notify_linger_seconds", linger_budget)
    monkeypatch.setattr(
        "hermes_cli.quiet_single_query.QUIET_BOT_DELIVERY_POLL_SECONDS", 0.05,
    )
    monkeypatch.setattr(process_registry, "wait_for_pending_completions", observe_real_wait)
    # The real registry wait and child exit event remain active. Only formatting/draining
    # the notification is omitted because the fake conversation engine has no LLM turn.
    monkeypatch.setattr(process_registry, "drain_notifications", lambda *args, **kwargs: [])
    monkeypatch.setattr(mailbox, "close_cli_consumer", admit_at_final_close)

    def run_conversation(*, user_message, conversation_history, **kwargs):
        calls.append({
            "message": user_message,
            "home": hermes_constants.get_hermes_home().resolve(),
            "author": kwargs.get("turn_author"),
        })
        if user_message == "original question":
            assert calls[-1]["home"] == home_b.resolve()
            assert mailbox.find_canonical_live_owner(home_b, include_cli=True) is not None
            return {
                "final_response": "original answer",
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "original answer"},
                ],
            }

        assert user_message == "message admitted at final close"
        assert calls[-1]["home"] == home_b.resolve()
        assert calls[-1]["author"] == author
        ticket = mailbox.read_delivery_result(home_b, delivery[0]["delivery_id"])
        assert ticket["status"] == "claimed"

        # Use the existing tracked local-process path, with a real child process and
        # a durable marker written only after the simulated background work completes.
        program = (
            "import pathlib,time; time.sleep(0.25); "
            f"pathlib.Path({str(marker)!r}).write_text(str(time.monotonic()))"
        )
        child = process_registry.spawn_local(
            f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}",
            cwd=str(tmp_path), task_id="chat-beta", owner_task_id="chat-beta",
            session_key="chat-beta", env_vars={"HERMES_HOME": str(home_b)},
        )
        child.notify_on_complete = True
        children.append(child)
        assert child.pid and child.process is not None
        assert not marker.exists()
        return {
            "final_response": "mailbox turn started child",
            "messages": conversation_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "mailbox turn started child"},
            ],
        }

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="chat-beta", _session_db=db)
    fake_cli = SimpleNamespace(
        agent=agent,
        conversation_history=[],
        session_id="chat-beta",
        _active_session_lease=lease,
        _release_active_session=lease.release,
    )
    home_token = hermes_constants.set_hermes_home_override(home_b)
    try:
        with pytest.raises(SystemExit) as exc:
            cli._run_quiet_single_query(fake_cli, "original question")
        assert exc.value.code == 0
        assert [call["message"] for call in calls] == [
            "original question", "message admitted at final close",
        ]
        assert len(linger_started) == 1, "the final drain must use the original linger deadline"
        assert len(children) == 1
        assert marker.is_file(), (
            "quiet CLI returned before its final-close mailbox turn's notify child completed; "
            f"waits={wait_records!r}"
        )
        marker_completed_at = float(marker.read_text())
        assert marker_completed_at <= linger_started[0] + linger_seconds
        assert children[0].exited and children[0]._completion_event.is_set()
        assert any(children[0].id in row["result"].get("completed", []) for row in wait_records), (
            "the real pending-completion wait did not observe the mailbox turn's child"
        )
        receipt = mailbox.read_delivery_result(home_b, delivery[0]["delivery_id"])
        assert receipt["status"] == "settled"
        assert receipt["reply"] == "mailbox turn started child"
        assert capsys.readouterr().out.strip() == "original answer"
        assert fake_cli._quiet_notify_linger_done is True

        # Exercise the actual one-shot finalizer. Its background-wait step must see that
        # the original quiet linger already covered the late child and must not wait anew.
        waits_before_finalize = len(wait_records)
        monkeypatch.setattr(cli, "_flush_one_shot_session_store", lambda _cli: None)
        monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _cli: None)
        monkeypatch.setattr(
            cli, "_run_cleanup",
            lambda **_kwargs: process_registry.kill_all(
                task_id="chat-beta", source="kill_all", consume_output=True,
            ),
        )
        cli._finalize_single_query(fake_cli)
        assert len(wait_records) == waits_before_finalize
        assert lease.released is True
    finally:
        process_registry.kill_all(task_id="chat-beta", source="kill_all", consume_output=True)
        lease.release()
        db.close()
        hermes_constants.reset_hermes_home_override(home_token)
