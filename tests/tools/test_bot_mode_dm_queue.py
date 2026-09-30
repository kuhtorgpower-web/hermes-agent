"""Quiet Bot Chat owners consume durable message_agent deliveries at idle boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
from hermes_cli.active_sessions import active_session_registry_snapshot, try_acquire_active_session
from hermes_state import SessionDB
from tools import bot_live_delivery as mailbox


def _cli_owner(home):
    home.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="chat-a", source="cli")
    db.set_session_title("chat-a", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat-a", surface="cli", config={}, registry_home=home,
        metadata={"live_session_id": "live-a"},
    )
    assert refusal is None and lease is not None
    owner = mailbox.advertise_cli_consumer(
        home, lease, session_id="chat-a", live_session_id="live-a",
    )
    assert owner is not None
    return db, lease, owner


def test_quiet_canonical_cli_consumes_mailbox_without_replacing_original_reply(
        tmp_path, monkeypatch, capsys):
    home = tmp_path / "profiles" / "codingpilot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TURN_AUTHOR", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="chat-a", source="cli")
    db.set_session_title("chat-a", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat-a", surface="cli", config={}, registry_home=home,
        metadata={"live_session_id": "live-a"},
    )
    assert refusal is None and lease is not None

    author = {"id": "bot:reviewpilot", "name": "Reviewpilot", "is_bot": True}
    receipts = []
    calls = []

    def run_conversation(*, user_message, conversation_history, **kwargs):
        calls.append((user_message, list(conversation_history), kwargs.get("turn_author")))
        if user_message == "original question":
            # The target is still busy with its first turn when another profile queues a DM.
            assert mailbox.find_canonical_live_owner(home) is None
            owner = mailbox.find_canonical_live_owner(home, include_cli=True)
            assert owner is not None, "quiet Bot Chat CLI did not advertise its active mailbox poller"
            receipts.append(mailbox.deliver_to_live_owner(
                home, owner, "queued teammate question", author=author,
            ))
            return {
                "final_response": "original answer",
                "messages": [{"role": "assistant", "content": "original answer"}],
            }
        assert user_message == "queued teammate question"
        return {
            "final_response": "mailbox reply",
            "messages": conversation_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "mailbox reply"},
            ],
        }

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="chat-a", _session_db=db)
    fake_cli = SimpleNamespace(
        agent=agent,
        conversation_history=[],
        session_id="chat-a",
        _active_session_lease=lease,
    )
    monkeypatch.setattr("hermes_cli.quiet_single_query.quiet_notify_linger_seconds", lambda: 0.0)

    try:
        with pytest.raises(SystemExit) as exc:
            cli._run_quiet_single_query(fake_cli, "original question")
        assert exc.value.code == 0

        receipt = mailbox.read_delivery_result(home, receipts[0]["delivery_id"])
        assert receipt["status"] == "settled"
        assert receipt["reply"] == "mailbox reply"
        assert calls[1][0] == "queued teammate question"
        assert calls[1][2] == author
        stdout = capsys.readouterr().out
        assert "original answer" in stdout
        assert "mailbox reply" not in stdout
        assert mailbox.find_canonical_live_owner(home, include_cli=True) is None
        assert lease.released is False
    finally:
        lease.release()
        db.close()


def test_cli_close_and_admission_share_the_mailbox_fence(tmp_path):
    home = tmp_path / "profiles" / "codingpilot"
    db, lease, owner = _cli_owner(home)
    try:
        first = mailbox.deliver_to_live_owner(
            home, owner, "accepted before close", delivery_id="a" * 32,
        )
        assert mailbox.close_cli_consumer(home, lease, live_session_id="live-a") is False
        current = mailbox.find_canonical_live_owner(home, include_cli=True)
        assert current is not None and current["lease_id"] == lease.lease_id

        # A pending long turn remains an accepting consumer while the owner drains it.
        second = mailbox.deliver_to_live_owner(
            home, current, "accepted while draining", delivery_id="b" * 32,
        )
        assert second["status"] == "queued"
        for expected in (first, second):
            claimed = mailbox.claim_pending_delivery(home, current)
            assert claimed["delivery_id"] == expected["delivery_id"]
            mailbox.complete_delivery(home, claimed["delivery_id"], status="settled", reply="ok")
        assert mailbox.close_cli_consumer(home, lease, live_session_id="live-a") is True

        with pytest.raises(mailbox.LiveOwnerNotAccepting):
            mailbox.deliver_to_live_owner(
                home, current, "late after close", delivery_id="c" * 32,
            )
        assert mailbox.read_delivery_result(home, "c" * 32) is None
        assert mailbox.find_canonical_live_owner(home) is None
        assert mailbox.find_canonical_live_owner(home, include_cli=True) is None
    finally:
        lease.release()
        db.close()


def test_cli_mailbox_follows_pending_receipt_through_compression(tmp_path):
    home = tmp_path / "profiles" / "codingpilot"
    db, lease, owner = _cli_owner(home)
    try:
        queued = mailbox.deliver_to_live_owner(home, owner, "before compression", delivery_id="d" * 32)
        db.end_session("chat-a", "compression")
        db.create_session(session_id="chat-b", source="cli", parent_session_id="chat-a")
        assert mailbox.sync_cli_consumer(
            home, lease, session_id="chat-b", live_session_id="live-a",
        )
        current = mailbox.find_canonical_live_owner(home, include_cli=True)
        assert current["session_id"] == "chat-b"
        assert current["live_session_id"] == "live-a"
        claimed = mailbox.claim_pending_delivery(home, current)
        assert claimed["delivery_id"] == queued["delivery_id"]
        assert claimed["session_id"] == "chat-a"
        mailbox.complete_delivery(home, claimed["delivery_id"], status="settled", reply="after compression")
        assert mailbox.close_cli_consumer(home, lease, live_session_id="live-a") is True
    finally:
        lease.release()
        db.close()


def test_cli_compression_sync_rejects_a_child_session_owned_by_desktop(tmp_path):
    from hermes_cli.active_sessions import transfer_active_session

    home = tmp_path / "profiles" / "codingpilot"
    db, lease, owner = _cli_owner(home)
    desktop_lease = None
    try:
        queued_before = mailbox.deliver_to_live_owner(
            home, owner, "queued before compression", delivery_id="e" * 32,
        )
        db.end_session("chat-a", "compression")
        db.create_session(session_id="chat-b", source="cli", parent_session_id="chat-a")

        # Before Desktop takes the new tip, the still-current CLI consumer can admit a
        # second ticket pinned to that child session.
        child_owner = mailbox.find_canonical_live_owner(home, include_cli=True)
        assert child_owner is not None and child_owner["session_id"] == "chat-b"
        queued_after = mailbox.deliver_to_live_owner(
            home, child_owner, "queued after compression", delivery_id="f" * 32,
        )
        desktop_lease, refusal = try_acquire_active_session(
            session_id="chat-b", surface="desktop", config={}, registry_home=home,
            metadata={"live_session_id": "desktop-b"}, track_liveness=True,
        )
        assert refusal is None and desktop_lease is not None
        before_rejected_transfer = active_session_registry_snapshot(home)
        assert not transfer_active_session(
            lease, session_id="chat-b", metadata={"live_session_id": "live-a"},
            require_exclusive=True,
        )
        assert active_session_registry_snapshot(home) == before_rejected_transfer
        assert lease.session_id == "chat-a"

        fake_cli = SimpleNamespace(
            session_id="chat-b", _active_session_lease=lease,
            _quiet_bot_live_delivery={"home": home, "live_session_id": "live-a"},
        )
        from hermes_cli.quiet_single_query import sync_quiet_cli_live_delivery

        assert sync_quiet_cli_live_delivery(fake_cli) is False
        assert fake_cli._quiet_bot_live_delivery is None
        assert lease.session_id == "chat-a", "failed sync must leave the original CLI slot in place"
        entries = mailbox.find_canonical_owner(home)
        assert entries["session_id"] == "chat-b"
        registry = active_session_registry_snapshot(home)
        assert sorted((row["session_id"], row["surface"]) for row in registry) == [
            ("chat-a", "cli"), ("chat-b", "desktop"),
        ]
        for record in (queued_before, queued_after):
            ticket = mailbox.read_delivery_result(home, record["delivery_id"])
            assert ticket["status"] == "cancelled"
        assert mailbox.find_canonical_live_owner(home, include_cli=True) is None
    finally:
        lease.release()
        if desktop_lease is not None:
            desktop_lease.release()
        db.close()


def test_quiet_flow_stops_before_notifications_after_compression_tip_is_taken(tmp_path, monkeypatch, capsys):
    home = tmp_path / "profiles" / "codingpilot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TURN_AUTHOR", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="chat-a", source="cli")
    db.set_session_title("chat-a", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat-a", surface="cli", config={}, registry_home=home,
        metadata={"live_session_id": "live-a"},
    )
    assert refusal is None and lease is not None
    desktop_lease = None
    ticket_ids = []
    model_calls = []
    drained_notifications = []

    def run_conversation(*, user_message, conversation_history, **kwargs):
        model_calls.append(user_message)
        assert user_message == "original question", "CLI ran a turn after Desktop took the tip"
        owner = mailbox.find_canonical_live_owner(home, include_cli=True)
        ticket = mailbox.deliver_to_live_owner(home, owner, "must not run after collision")
        ticket_ids.append(ticket["delivery_id"])
        db.end_session("chat-a", "compression")
        db.create_session(session_id="chat-b", source="cli", parent_session_id="chat-a")
        nonlocal desktop_lease
        desktop_lease, refusal = try_acquire_active_session(
            session_id="chat-b", surface="desktop", config={}, registry_home=home,
            metadata={"live_session_id": "desktop-b"}, track_liveness=True,
        )
        assert refusal is None and desktop_lease is not None
        agent.session_id = "chat-b"
        return {
            "final_response": "original answer",
            "messages": [{"role": "assistant", "content": "original answer"}],
        }

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="chat-a", _session_db=db)
    fake_cli = SimpleNamespace(
        agent=agent, conversation_history=[], session_id="chat-a", _active_session_lease=lease,
    )

    def wait_for_notification(*args, **kwargs):
        return {"waited": ["pending-notification"], "completed": [], "timed_out": False}

    def drain_pending_notification(*args, **kwargs):
        drained_notifications.append(True)
        return [({"type": "completion"}, "notification follow-up")]

    monkeypatch.setattr(
        "tools.process_registry.process_registry.wait_for_pending_completions",
        wait_for_notification,
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry.drain_notifications",
        drain_pending_notification,
    )
    monkeypatch.setattr("hermes_cli.quiet_single_query.quiet_notify_linger_seconds", lambda: 1.0)
    try:
        with pytest.raises(SystemExit) as exc:
            cli._run_quiet_single_query(fake_cli, "original question")
        assert exc.value.code == 0
        assert model_calls == ["original question"]
        assert drained_notifications == [], "the quiet loop must stop before dispatching pending notifications"
        assert mailbox.read_delivery_result(home, ticket_ids[0])["status"] == "cancelled"
        assert lease.session_id == "chat-a"
        registry = active_session_registry_snapshot(home)
        assert sorted((row["session_id"], row["surface"]) for row in registry) == [
            ("chat-a", "cli"), ("chat-b", "desktop"),
        ]
        assert not registry[0]["metadata"].get("bot_live_delivery_consumer")
        assert mailbox.find_canonical_live_owner(home, include_cli=True) is None
        assert "original answer" in capsys.readouterr().out
    finally:
        lease.release()
        if desktop_lease is not None:
            desktop_lease.release()
        db.close()


def test_turn_lock_busy_falls_back_to_same_live_mailbox_delivery(tmp_path, monkeypatch):
    from tools import bot_mode_dm, bot_relay
    from tools.bot_relay import acquire_turn_lock

    home = tmp_path / "profiles" / "codingpilot"
    db, lease, owner = _cli_owner(home)
    dm_file = tmp_path / "dm.txt"
    dm_file.write_text("queue after lock timeout", encoding="utf-8")
    attempts = []
    real_admit = bot_mode_dm._admit_live_dm
    author = {"id": "bot:reviewpilot", "name": "Reviewpilot", "is_bot": True}

    def admit(profile_home, payload, payload_author=None):
        attempts.append(payload)
        if len(attempts) == 1:
            return None  # Simulate the initial owner lookup racing advertisement.
        return real_admit(profile_home, payload, payload_author)

    def wait(home_path, delivery_id, *, dm_file=None):
        receipt = mailbox.read_delivery_result(home_path, delivery_id)
        assert receipt["status"] == "queued"
        assert Path(dm_file).exists()
        return 0

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(bot_mode_dm, "_admit_live_dm", admit)
    monkeypatch.setattr(bot_mode_dm, "_wait_live_dm", wait)
    monkeypatch.setattr(bot_relay, "turn_wait_seconds", lambda: 0.02)
    try:
        with acquire_turn_lock(tmp_path, "codingpilot", timeout_seconds=5):
            rc = bot_mode_dm._run_delivery(
                ["hermes", "-p", "codingpilot", "chat"], str(dm_file),
                stdin_file=False, profile_home=home, author=author,
            )
        assert rc == 0
        assert len(attempts) == 2 and attempts == [str(dm_file), str(dm_file)]
        record = mailbox.read_delivery_result(home, bot_mode_dm._dm_delivery_id(dm_file))
        assert record["status"] == "queued" and record["author"] == author
        assert dm_file.exists() and Path(str(dm_file) + ".live.json").exists()
    finally:
        lease.release()
        db.close()


@pytest.mark.parametrize("interrupt_mode", ["raise", "result"])
def test_interrupted_quiet_turn_cancels_queued_receipts_without_running_them(
        tmp_path, monkeypatch, interrupt_mode):
    home = tmp_path / "profiles" / "codingpilot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TURN_AUTHOR", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="chat-a", source="cli")
    db.set_session_title("chat-a", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat-a", surface="cli", config={}, registry_home=home,
        metadata={"live_session_id": "live-a"},
    )
    assert refusal is None and lease is not None
    delivery_ids = []
    run_calls = []

    def run_conversation(*, user_message, conversation_history, **kwargs):
        run_calls.append(user_message)
        owner = mailbox.find_canonical_live_owner(home, include_cli=True)
        delivery = mailbox.deliver_to_live_owner(home, owner, "must be cancelled on stop")
        delivery_ids.append(delivery["delivery_id"])
        if interrupt_mode == "raise":
            raise KeyboardInterrupt
        return {"interrupted": True, "final_response": "", "messages": []}

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="chat-a", _session_db=db)
    fake_cli = SimpleNamespace(
        agent=agent, conversation_history=[], session_id="chat-a", _active_session_lease=lease,
    )
    monkeypatch.setattr(cli, "_emit_interrupted_session_end", lambda *a, **k: None)
    try:
        with pytest.raises(SystemExit) as exc:
            cli._run_quiet_single_query(fake_cli, "interrupt me")
        assert exc.value.code == 130
        assert run_calls == ["interrupt me"]
        assert mailbox.read_delivery_result(home, delivery_ids[0])["status"] == "cancelled"
        assert mailbox.find_canonical_live_owner(home, include_cli=True) is None
    finally:
        lease.release()
        db.close()


def test_interrupted_mailbox_turn_cancels_remaining_queue_without_another_turn(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "codingpilot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TURN_AUTHOR", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="chat-a", source="cli")
    db.set_session_title("chat-a", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat-a", surface="cli", config={}, registry_home=home,
        metadata={"live_session_id": "live-a"},
    )
    assert refusal is None and lease is not None
    delivery_ids = []
    run_calls = []

    def run_conversation(*, user_message, conversation_history, **kwargs):
        run_calls.append(user_message)
        if user_message == "original question":
            owner = mailbox.find_canonical_live_owner(home, include_cli=True)
            for text in ("first mailbox turn", "second mailbox turn"):
                delivery = mailbox.deliver_to_live_owner(home, owner, text)
                delivery_ids.append(delivery["delivery_id"])
            return {
                "final_response": "original answer",
                "messages": [{"role": "assistant", "content": "original answer"}],
            }
        raise KeyboardInterrupt

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="chat-a", _session_db=db)
    fake_cli = SimpleNamespace(
        agent=agent, conversation_history=[], session_id="chat-a", _active_session_lease=lease,
    )
    try:
        with pytest.raises(SystemExit) as exc:
            cli._run_quiet_single_query(fake_cli, "original question")
        assert exc.value.code == 130
        assert run_calls == ["original question", "first mailbox turn"]
        assert [mailbox.read_delivery_result(home, key)["status"] for key in delivery_ids] == [
            "cancelled", "cancelled",
        ]
        assert mailbox.find_canonical_live_owner(home, include_cli=True) is None
    finally:
        lease.release()
        db.close()


def test_interrupted_notify_followup_cancels_pending_mailbox_without_another_turn(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "codingpilot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TURN_AUTHOR", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="chat-a", source="cli")
    db.set_session_title("chat-a", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat-a", surface="cli", config={}, registry_home=home,
        metadata={"live_session_id": "live-a"},
    )
    assert refusal is None and lease is not None
    run_calls = []
    delivery_ids = []

    def run_conversation(*, user_message, conversation_history, **kwargs):
        run_calls.append(user_message)
        if user_message == "original question":
            return {
                "final_response": "original answer",
                "messages": [{"role": "assistant", "content": "original answer"}],
            }
        if user_message == "notification follow-up":
            owner = mailbox.find_canonical_live_owner(home, include_cli=True)
            delivery = mailbox.deliver_to_live_owner(home, owner, "must not run after stop")
            delivery_ids.append(delivery["delivery_id"])
            return {"interrupted": True, "final_response": "", "messages": []}
        raise AssertionError(f"unexpected model turn: {user_message}")

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="chat-a", _session_db=db)
    fake_cli = SimpleNamespace(
        agent=agent, conversation_history=[], session_id="chat-a", _active_session_lease=lease,
    )
    monkeypatch.setattr(
        "hermes_cli.quiet_single_query.continue_quiet_notify_completions",
        lambda session_id, run_turn, **kwargs: run_turn("notification follow-up"),
    )
    try:
        with pytest.raises(SystemExit) as exc:
            cli._run_quiet_single_query(fake_cli, "original question")
        assert exc.value.code == 130
        assert run_calls == ["original question", "notification follow-up"]
        assert mailbox.read_delivery_result(home, delivery_ids[0])["status"] == "cancelled"
        assert mailbox.find_canonical_live_owner(home, include_cli=True) is None
    finally:
        lease.release()
        db.close()
