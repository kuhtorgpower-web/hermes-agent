"""Durable, at-most-once handoff to an existing Bot Chat owner.

Adapted from FalconOrtiz's live-owner mailbox (#101564). A single private
record advances queued -> claimed -> terminal under a process-shared lock.
Claims never expire: a crashed consumer leaves an inspectable unknown outcome,
not permission to execute the same input again. Receipts are permanent.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager

from utils import atomic_json_write, atomic_write_text, fsync_directory
from pathlib import Path
from typing import Any, Callable

from hermes_cli.active_sessions import _FileLock

log = logging.getLogger(__name__)

DELIVERY_DIR_NAME = "bot_live_delivery"
_SEQUENCE_FILE = ".sequence"
_OWNER_KEYS = ("profile_home", "session_id", "lease_id", "live_session_id")
_TERMINAL = frozenset({"settled", "failed", "cancelled", "ambiguous"})
_CLI_CONSUMER_KIND = "cli"
_CLI_CONSUMER_KIND_KEY = "bot_live_delivery_consumer_kind"


class LiveOwnerNotAccepting(RuntimeError):
    """The pinned CLI owner stopped accepting new mailbox deliveries before admission."""


def find_canonical_owner(profile_home: Path | str) -> dict[str, Any] | None:
    """Return the exact Bot Chat tip's lease, including unsupported CLI owners."""
    from hermes_cli.active_sessions import active_session_registry_snapshot
    from hermes_state import SessionDB

    home = Path(profile_home).resolve()
    if not (home / "state.db").is_file():
        return None
    db = SessionDB(db_path=home / "state.db", read_only=True)
    try:
        row = db.get_session_by_title("Bot Chat")
        session_id = db.get_compression_tip(row["id"]) if row else None
    finally:
        db.close()
    if not session_id:
        return None
    for entry in active_session_registry_snapshot(registry_home=home):
        if entry["session_id"] == session_id:
            return {**entry, "profile_home": str(home)}
    return None


def find_canonical_live_owner(
    profile_home: Path | str, *, include_cli: bool = False,
) -> dict[str, Any] | None:
    """Find an advertised owner; CLI owners are opt-in for the local Bot Chat route only."""
    entry = find_canonical_owner(profile_home)
    meta = (entry or {}).get("metadata") or {}
    if meta.get(_CLI_CONSUMER_KIND_KEY) == _CLI_CONSUMER_KIND and not include_cli:
        return None
    # A quiet CLI can compress its canonical session before its caller updates the lease.
    # Follow that same lease through the compression chain for the short handoff window.
    if entry is None and include_cli:
        entry = _current_cli_consumer(profile_home)
    meta = (entry or {}).get("metadata") or {}
    if entry and meta.get("bot_live_delivery_consumer") is True and meta.get("live_session_id"):
        result = {key: entry[key] for key in ("profile_home", "session_id", "lease_id")} | {
            "live_session_id": meta["live_session_id"]}
        kind = meta.get(_CLI_CONSUMER_KIND_KEY)
        if kind == _CLI_CONSUMER_KIND:
            result["consumer_kind"] = kind
        return result
    return None


def _canonical_bot_chat_tip(profile_home: Path | str):
    from hermes_state import SessionDB

    home = Path(profile_home).resolve()
    if not (home / "state.db").is_file():
        return None, None
    db = SessionDB(db_path=home / "state.db", read_only=True)
    try:
        row = db.get_session_by_title("Bot Chat")
        return (db.get_compression_tip(row["id"]) if row else None), db
    except BaseException:
        db.close()
        raise


def _current_cli_consumer(profile_home: Path | str, lease_id: str | None = None) -> dict[str, Any] | None:
    """Find the advertised CLI lease for the canonical Bot Chat tip, including a pending compression transfer."""
    from hermes_cli.active_sessions import active_session_registry_snapshot

    home = Path(profile_home).resolve()
    tip, db = _canonical_bot_chat_tip(home)
    if db is None:
        return None
    try:
        if not tip:
            return None
        for entry in active_session_registry_snapshot(registry_home=home):
            if lease_id is not None and entry.get("lease_id") != lease_id:
                continue
            meta = entry.get("metadata") or {}
            if (meta.get("bot_live_delivery_consumer") is not True
                    or meta.get(_CLI_CONSUMER_KIND_KEY) != _CLI_CONSUMER_KIND
                    or not meta.get("live_session_id")):
                continue
            if db.get_compression_tip(entry.get("session_id")) == tip:
                return {**entry, "profile_home": str(home), "session_id": tip,
                        "live_session_id": meta["live_session_id"]}
    finally:
        db.close()
    return None


def _cli_consumer_metadata(metadata: dict[str, Any], live_session_id: str) -> dict[str, Any]:
    updated = dict(metadata)
    updated.update(
        live_session_id=live_session_id,
        bot_live_delivery_consumer=True,
        **{_CLI_CONSUMER_KIND_KEY: _CLI_CONSUMER_KIND},
    )
    return updated


def advertise_cli_consumer(
    profile_home: Path | str, lease: Any, *, session_id: str, live_session_id: str,
) -> dict[str, Any] | None:
    """Advertise the quiet CLI poller only while it owns the canonical Bot Chat lease."""
    from hermes_cli.active_sessions import active_session_registry_snapshot, transfer_active_session

    home = Path(profile_home).resolve()
    if (not session_id or not live_session_id or lease is None
            or getattr(lease, "released", True) or not getattr(lease, "enabled", False)):
        return None
    with _locked(home):
        tip, db = _canonical_bot_chat_tip(home)
        if db is None:
            return None
        try:
            if tip != str(session_id):
                return None
        finally:
            db.close()
        entry = next((item for item in active_session_registry_snapshot(registry_home=home)
                      if item.get("lease_id") == lease.lease_id), None)
        if entry is None or entry.get("session_id") != str(session_id):
            return None
        metadata = _cli_consumer_metadata(entry.get("metadata") or {}, live_session_id)
        if not transfer_active_session(
            lease, session_id=str(session_id), metadata=metadata, require_exclusive=True,
        ):
            return None
        return {
            "profile_home": str(home), "session_id": str(session_id),
            "lease_id": lease.lease_id, "live_session_id": live_session_id,
            "consumer_kind": _CLI_CONSUMER_KIND,
        }


def sync_cli_consumer(
    profile_home: Path | str, lease: Any, *, session_id: str, live_session_id: str,
) -> bool:
    """Keep the advertised lease on the canonical compression tip with its stable live id."""
    from hermes_cli.active_sessions import transfer_active_session

    home = Path(profile_home).resolve()
    if (not session_id or not live_session_id or lease is None
            or getattr(lease, "released", True) or not getattr(lease, "enabled", False)):
        return False
    with _locked(home):
        current = _current_cli_consumer(home, lease.lease_id)
        if (current is None or current["metadata"].get("live_session_id") != live_session_id
                or current["session_id"] != str(session_id)):
            return False
        metadata = _cli_consumer_metadata(current.get("metadata") or {}, live_session_id)
        return transfer_active_session(
            lease, session_id=current["session_id"], metadata=metadata, require_exclusive=True,
        )


def close_cli_consumer(profile_home: Path | str, lease: Any, *, live_session_id: str) -> bool:
    """Withdraw the CLI consumer under the admission lock once no accepted work is pending."""
    from hermes_cli.active_sessions import transfer_active_session

    home = Path(profile_home).resolve()
    if not live_session_id or lease is None or getattr(lease, "released", True):
        return True
    with _locked(home) as root:
        current = _current_cli_consumer(home, lease.lease_id)
        if current is None or current["metadata"].get("live_session_id") != live_session_id:
            return True
        pending = any(
            (record := _scan_read(path)) is not None
            and record["status"] in _PENDING
            and _matches(home, record, current)
            for path in root.glob("*.json")
        )
        metadata = dict(current.get("metadata") or {})
        if pending:
            return False
        metadata.pop(_CLI_CONSUMER_KIND_KEY, None)
        metadata.pop("bot_live_delivery_consumer", None)
        metadata["live_session_id"] = live_session_id
        if not transfer_active_session(
            lease, session_id=current["session_id"], metadata=metadata, require_exclusive=True,
        ):
            return False
        return True


def abort_cli_consumer(
    profile_home: Path | str, lease: Any, *, live_session_id: str, reason: str = "quiet CLI owner stopped",
) -> bool:
    """On interruption, cancel only unclaimed tickets and withdraw this one-shot consumer."""
    from hermes_cli.active_sessions import transfer_active_session

    home = Path(profile_home).resolve()
    if not live_session_id or lease is None or getattr(lease, "released", True):
        return True
    with _locked(home) as root:
        current = _current_cli_consumer(home, lease.lease_id)
        if current is None or current.get("live_session_id") != live_session_id:
            return True
        for path in root.glob("*.json"):
            record = _scan_read(path)
            if record is None or record["status"] != "queued" or not _matches(home, record, current):
                continue
            record.update(status="cancelled", reply="", error=reason, reason="cancelled",
                          completed_at=time.time_ns())
            _write(path, record)
        metadata = dict(current.get("metadata") or {})
        metadata.pop(_CLI_CONSUMER_KIND_KEY, None)
        metadata.pop("bot_live_delivery_consumer", None)
        metadata["live_session_id"] = live_session_id
        # The CLI lease may still be on an older tip when another surface has already
        # acquired the compression child. Withdraw in place; never move this lease onto
        # the competing surface's session.
        return transfer_active_session(lease, session_id=lease.session_id, metadata=metadata)


def _owner(home: Path | str, owner: dict[str, Any]) -> dict[str, str]:
    pinned = {key: owner.get(key) for key in _OWNER_KEYS}
    if not all(isinstance(value, str) and value for value in pinned.values()):
        raise ValueError("owner requires profile_home, session_id, lease_id and live_session_id")
    if pinned["profile_home"] != str(Path(home).resolve()):
        raise ValueError("owner belongs to a different profile home")
    return pinned


def _delivery_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32,64}", value) is None:
        raise ValueError("delivery id must be 32 to 64 lowercase hex characters")
    return value


def _root(home: Path | str) -> Path:
    return Path(home).resolve() / "runtime" / DELIVERY_DIR_NAME


def has_mailbox(profile_home: Path | str) -> bool:
    """Whether any delivery was ever admitted for this profile (the mailbox directory is created on
    first admission only). A cheap pre-check for pollers: no mailbox means nothing to claim, so the
    owner lookup — a state.db open plus the exclusive active-session registry lock — can be skipped."""
    return _root(profile_home).is_dir()


@contextmanager
def _locked(home: Path | str):
    root = _root(home)
    created = not root.is_dir()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    if created:
        # Only a fresh mailbox dir needs its parents durably linked; the live
        # poller re-enters this lock twice a second per profile, and two
        # directory fsyncs per idle poll was measurable disk churn for nothing.
        fsync_directory(root.parent)
        fsync_directory(root.parent.parent)
    lock = root / ".lock"
    fd = os.open(lock, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    with _FileLock(lock):
        yield root


def _read(path: Path) -> dict[str, Any] | None:
    """Exact-id read: absent → None; unreadable or not a JSON object → raises (callers fail closed)."""
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"ticket {path.name} is not a JSON object ({type(record).__name__})")
    return record


# Tickets already reported unreadable by this process. The live poller rescans the
# dir twice a second, so a persistent bad ticket is WARNING once and DEBUG after.
_warned_unreadable: set[Path] = set()


def _ticket_shape_error(path: Path, record: dict[str, Any]) -> str | None:
    """Why a parsed ticket is unusable by the scans, or None when it is well-formed.

    A ticket that parses as JSON but lost a field (truncated rewrite, foreign
    writer, hand edit) used to raise KeyError/TypeError out of the sequence
    scan and the claim sweep — wedging admission and delivery for the whole
    profile exactly like corrupt JSON did before ``_scan_read`` existed.
    """
    owner = record.get("owner")
    created_at, sequence = record.get("created_at"), record.get("sequence", record.get("created_at"))
    if record.get("delivery_id") != path.stem or record.get("id") != path.stem:
        return "id does not match filename"
    status = record.get("status")
    if not isinstance(status, str) or status not in ({"queued", "claimed"} | _TERMINAL):
        return f"unknown status {status!r}"
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (created_at, sequence)):
        return "created_at/sequence are not integers"
    if not isinstance(owner, dict) or not all(isinstance(owner.get(k), str) and owner[k] for k in _OWNER_KEYS):
        return "owner pin is incomplete"
    return None


def _scan_read(path: Path) -> dict[str, Any] | None:
    """Bulk-scan variant: one unreadable or malformed ticket must not wedge the whole dir.

    Directory scans (sequence high-water mark, queued-claim sweep) may only
    treat a file as absent when it is provably absent; an unreadable ticket
    degrades to "that one delivery is uninspectable" with a warning.
    Exact-id reads (admission idempotency, completion, result lookup) keep
    using _read so a permission error still fails closed instead of
    licensing an overwrite of a possibly-live receipt.
    """
    try:
        record = _read(path)
        problem = None if record is None else _ticket_shape_error(path, record)
    except (OSError, ValueError) as exc:  # ValueError: corrupt JSON and invalid UTF-8 alike
        record, problem = None, str(exc)
    if problem is not None:
        level = logging.DEBUG if path in _warned_unreadable else logging.WARNING
        _warned_unreadable.add(path)
        log.log(level, "bot_live_delivery: skipping unreadable ticket %s (%s)", path.name, problem)
        return None
    _warned_unreadable.discard(path)
    return record


def _next_sequence(root: Path) -> int:
    """Allocate the next admission sequence under the dir lock.

    The high-water mark lives in a counter file beside the tickets, so a ticket
    the scan cannot read does not drop its sequence and hand a later admission
    a duplicate or lower one. Readable tickets still bootstrap dirs written
    before the counter existed. Wall time can roll back; sequences never do.
    """
    counter = root / _SEQUENCE_FILE
    try:
        persisted = int(counter.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        persisted = 0
    scanned = max((record.get("sequence", record["created_at"])
                   for candidate in root.glob("*.json")
                   if (record := _scan_read(candidate)) is not None), default=0)
    sequence = max(persisted, scanned) + 1
    atomic_write_text(counter, str(sequence), mode=0o600, fsync_dir=True)
    return sequence


def _write(path: Path, record: dict[str, Any]) -> None:
    atomic_json_write(path, record, indent=None, sort_keys=True, fsync_dir=True, mode=0o600)


def deliver_to_live_owner(
    profile_home: Path | str, owner: dict[str, Any], message: str,
    *, delivery_id: str | None = None, author: dict[str, Any] | None = None,
    notification_category: str = "result",
) -> dict[str, Any]:
    """Return durable admission immediately, without waiting for the owner.

    Retry with the same id AND pinned owner/message to inspect the existing
    state. Reusing an id with a different payload is an error, never an overwrite.
    """
    pinned = _owner(profile_home, owner)
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    key = _delivery_id(delivery_id if delivery_id is not None else uuid.uuid4().hex)
    with _locked(profile_home) as root:
        path = root / f"{key}.json"
        existing = _read(path)
        if existing is not None:
            if (existing["owner"] != pinned or existing["message"] != message or existing.get("author") != author
                    or existing.get("notification_category", "result") != notification_category):
                raise ValueError("delivery id already belongs to a different payload")
            return existing
        if owner.get("consumer_kind") == _CLI_CONSUMER_KIND:
            current = _current_cli_consumer(profile_home, pinned["lease_id"])
            if (current is None
                    or current.get("metadata", {}).get("live_session_id") != pinned["live_session_id"]
                    or current.get("metadata", {}).get(_CLI_CONSUMER_KIND_KEY) != _CLI_CONSUMER_KIND):
                raise LiveOwnerNotAccepting("the Bot Chat CLI owner is no longer accepting new deliveries")
            if pinned["session_id"] != current["session_id"]:
                from hermes_state import SessionDB

                db = SessionDB(db_path=Path(profile_home) / "state.db", read_only=True)
                try:
                    if db.get_compression_tip(pinned["session_id"]) != current["session_id"]:
                        raise LiveOwnerNotAccepting("the pinned Bot Chat CLI owner is no longer canonical")
                finally:
                    db.close()
        record = dict(delivery_id=key, id=key, owner=pinned, **pinned,
                      message=message, status="queued", created_at=time.time_ns(),
                      sequence=_next_sequence(root), **({"author": dict(author)} if author else {}))
        if notification_category == "diagnostic":
            record["notification_category"] = notification_category
        _write(path, record)
        return record


def discard_unadmitted_intent(
    profile_home: Path | str, delivery_id: str, intent_path: Path | str, expected_intent: dict[str, Any],
) -> bool:
    """Remove only the exact intent whose ticket is still provably absent under the mailbox lock."""
    key = _delivery_id(delivery_id)
    path = Path(intent_path)
    with _locked(profile_home) as root:
        if _read(root / f"{key}.json") is not None:
            return False
        try:
            current = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return True
        except (OSError, ValueError):
            return False
        if current != expected_intent:
            return False
        path.unlink()
        from utils import fsync_directory

        fsync_directory(path.parent)
        return True


def _matches(home: Path | str, record: dict, owner: dict) -> bool:
    pinned = record["owner"]
    if any(pinned[key] != owner[key] for key in ("profile_home", "lease_id", "live_session_id")):
        return False
    if pinned["session_id"] == owner["session_id"]:
        return True
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(home) / "state.db", read_only=True)
    try:
        return db.get_compression_tip(pinned["session_id"]) == owner["session_id"]
    finally:
        db.close()


def claim_pending_delivery(
    profile_home: Path | str, owner: dict[str, Any],
) -> dict[str, Any] | None:
    """Claim oldest matching input exactly once; caller supplies its current lease.

    A lease transfer across compression is accepted only along the original
    stored session's compression chain. A new lease/live session cannot steal it.
    Caller must hold its normal turn-admission guard before invoking this.
    """
    current = _owner(profile_home, owner)
    if not _root(profile_home).is_dir():
        return None
    with _locked(profile_home) as root:
        pending = []
        for path in root.glob("*.json"):
            record = _scan_read(path)
            if record is not None and record["status"] == "queued" and _matches(profile_home, record, current):
                pending.append(record)
        if not pending:
            return None
        record = min(pending, key=lambda item: (
            item.get("sequence", item["created_at"]), item["delivery_id"]))
        record.update(status="claimed", claimed_at=time.time_ns())
        _write(root / f"{record['delivery_id']}.json", record)
        return record


def complete_delivery(
    profile_home: Path | str, delivery_id: str, *, status: str,
    reply: str = "", error: str = "", reason: str = "",
) -> dict[str, Any]:
    """Persist an immutable terminal receipt; duplicate identical completion is safe."""
    key = _delivery_id(delivery_id)
    if status not in _TERMINAL:
        raise ValueError("invalid terminal delivery status")
    outcome = dict(status=status, reply=reply, error=error, reason=reason)
    with _locked(profile_home) as root:
        path = root / f"{key}.json"
        record = _read(path)
        if record is None:
            raise FileNotFoundError(f"delivery not found: {key}")
        if record["status"] in _TERMINAL:
            if any(record.get(k) != v for k, v in outcome.items()):
                raise ValueError("delivery already has a different terminal receipt")
            return record
        if record["status"] != "claimed":
            raise ValueError("delivery must be claimed before completion")
        record.update(outcome, completed_at=time.time_ns())
        _write(path, record)
        return record


def read_delivery_result(profile_home: Path | str, delivery_id: str) -> dict[str, Any] | None:
    """Read admission/claim/terminal state without waiting or deleting its receipt."""
    return _read(_root(profile_home) / f"{_delivery_id(delivery_id)}.json")


_PENDING = ("queued", "claimed")
_POLL_SECONDS = 0.5


def await_delivery(
    profile_home: Path | str, delivery_id: str, timeout: float | None,
    *, should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """Poll a receipt until the owner settles it, ``timeout`` lapses, or ``should_stop`` says so.

    Every transport that hands a turn to a live Bot Chat owner (local ``message_agent``, the
    Desktop relay, ``hermes peer dm`` and ``hermes peer run``) waits on the same receipt; keeping
    the loop here is what stops the lanes drifting (one lane returned a receipt sentence instead
    of the reply, two never waited at all). Returns the last record read — still pending when the
    budget lapsed, None when the receipt was never readable.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        record = read_delivery_result(profile_home, delivery_id)
        if record is None or record["status"] not in _PENDING:
            return record
        if should_stop is not None and should_stop():
            return record
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return record
        time.sleep(_POLL_SECONDS if remaining is None else min(_POLL_SECONDS, remaining))


async def await_delivery_async(
    profile_home: Path | str, delivery_id: str, timeout: float | None,
    *, should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """``await_delivery`` for an event loop: never blocks a worker thread for the whole budget."""
    import asyncio

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        record = await asyncio.to_thread(read_delivery_result, profile_home, delivery_id)
        if record is None or record["status"] not in _PENDING:
            return record
        if should_stop is not None and should_stop():
            return record
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return record
        await asyncio.sleep(_POLL_SECONDS if remaining is None else min(_POLL_SECONDS, remaining))
