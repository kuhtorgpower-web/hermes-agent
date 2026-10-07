"""Profile-owned delegation caps constrain real child tool definitions."""

import json
from types import SimpleNamespace

import pytest

from agent.delegation_context import delegated_child_context
import hermes_cli.config as hc
import hermes_yaml as yaml
from model_tools import get_tool_definitions
from tools.delegate_tool_toolsets import _resolve_child_toolsets


def _configure(home, delegation):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"delegation": delegation, "model": {"supports_vision": True}}), encoding="utf-8",
    )
    hc._LOAD_CONFIG_CACHE.clear()


def _names(enabled, disabled):
    from agent.auxiliary_client import reset_runtime_main, set_runtime_main
    token = set_runtime_main("openai", "test-vision-model")
    try:
        with delegated_child_context():
            return {td["function"]["name"] for td in get_tool_definitions(
                enabled_toolsets=enabled, disabled_toolsets=disabled,
                quiet_mode=True, skip_tool_search_assembly=True,
            )}
    finally:
        reset_runtime_main(token)


def test_profile_cap_bounds_inheritance_explicit_requests_and_orchestrators(tmp_path, monkeypatch):
    """Cap applies after MCP inheritance and role grants; neither widens it."""
    home = tmp_path / "capped"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    _configure(home, {"child_toolsets": ["web", "vision"], "inherit_mcp_toolsets": True})
    expected = _names(["web", "vision"], [])
    assert "web_search" in expected
    assert "vision_analyze" in expected
    parents = [
        ["web", "vision", "terminal", "file", "skills", "code_execution", "delegation", "messaging", "mcp-test"],
        ["hermes-cli", "mcp-test"],
    ]
    for parent_enabled in parents:
        parent = SimpleNamespace(enabled_toolsets=parent_enabled, disabled_toolsets=[])
        before = list(parent_enabled)
        for request in (None, ["web", "vision", "terminal", "file"], ["hermes-cli"], ["terminal"]):
            for role in ("leaf", "orchestrator"):
                enabled, disabled = _resolve_child_toolsets(parent, request, role)
                actual = _names(enabled, disabled)
                assert actual <= expected
                if request != ["terminal"]:
                    assert actual == expected
                else:
                    assert actual == set()
                assert not {"terminal", "read_file", "write_file", "patch", "skill_manage", "execute_code",
                            "plugins_manage", "send_message", "message_agent", "delegate_task"} & actual
                assert not any(name.startswith("mcp-") for name in enabled)
        assert parent_enabled == before

    # Parent disables still apply inside the cap.
    parent = SimpleNamespace(enabled_toolsets=["hermes-cli"], disabled_toolsets=["vision"])
    enabled, disabled = _resolve_child_toolsets(parent, None, "orchestrator")
    assert "vision" in disabled
    assert "vision_analyze" not in _names(enabled, disabled)
    assert "delegation" in disabled

    # No capability absent from the parent is granted by a profile cap.
    parent = SimpleNamespace(enabled_toolsets=["web"], disabled_toolsets=[])
    enabled, disabled = _resolve_child_toolsets(parent, None, "leaf")
    assert enabled == ["web"]

    # message_agent is injected outside the registry, through a canonical-chat gate.
    from tools.bot_mode_dm import ensure_message_agent_tool, message_agent_tool
    peer = home / "profiles" / "researcher"
    peer.mkdir(parents=True)
    (peer / "profile.yaml").write_text("ui_meta:\n  hermes-bots:\n    shape: cloud\n", encoding="utf-8")
    child = SimpleNamespace(tools=[], valid_tool_names=set(), _session_title_hint="Subagent: research",
                            _session_db=SimpleNamespace(db_path=str(home / "state.db")))
    assert ensure_message_agent_tool(child) is False
    result = json.loads(message_agent_tool(target="researcher", message="forged child call", agent=child))
    assert result["error"]
    assert child.tools == []


def test_cap_empty_invalid_and_profile_switches_fail_closed(tmp_path, monkeypatch):
    """Missing cap preserves defaults; [] and malformed values cannot widen children."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from agent.secret_scope import is_multiplex_active, reset_secret_scope, set_multiplex_active, set_secret_scope

    home_a, home_b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    parent = SimpleNamespace(enabled_toolsets=["hermes-cli"], disabled_toolsets=[])
    _configure(home_a, {"child_toolsets": ["web", "vision"]})
    _configure(home_b, {})
    previous_multiplex = is_multiplex_active()
    home_token = set_hermes_home_override(home_a)
    secret_token = set_secret_scope({}, profile_home=str(home_a))
    set_multiplex_active(True)
    try:
        a = _resolve_child_toolsets(parent, None, "leaf")
        token = set_hermes_home_override(home_b)
        b_secret = set_secret_scope({}, profile_home=str(home_b))
        try:
            b = _resolve_child_toolsets(parent, None, "leaf")
            assert "hermes-cli" in b[0]
            assert "delegation" in b[1]
            simple = SimpleNamespace(enabled_toolsets=["terminal", "web"], disabled_toolsets=[])
            assert "terminal" in _names(*_resolve_child_toolsets(simple, None, "leaf"))
        finally:
            reset_secret_scope(b_secret)
            reset_hermes_home_override(token)
        assert _resolve_child_toolsets(parent, None, "leaf") == a
        assert "terminal" not in _names(*a)
    finally:
        set_multiplex_active(previous_multiplex)
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

    _configure(home_a, {"child_toolsets": []})
    for request in (None, ["hermes-cli"], []):
        for role in ("leaf", "orchestrator"):
            enabled, disabled = _resolve_child_toolsets(parent, request, role)
            assert enabled == []
            assert _names(enabled, disabled) == set()

    for invalid in (None, "web", False, {}, ["web", 3], ["unknown-child-toolset"], [""]):
        _configure(home_a, {"child_toolsets": invalid})
        with pytest.raises(ValueError, match="delegation.child_toolsets"):
            _resolve_child_toolsets(parent, None, "leaf")
