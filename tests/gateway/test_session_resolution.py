from __future__ import annotations

from polar.gateway.session import SessionRegistry, resolve_session_id


def test_registered_api_key_session_wins_over_client_session_id() -> None:
    registry = SessionRegistry()
    registry.register("sk-polar-abc", task_id="t1", registered=True)

    resolved = resolve_session_id(
        registry,
        {"Authorization": "Bearer sk-polar-abc", "x-session-id": "ses_opencode_internal"},
        {},
    )

    assert resolved == "sk-polar-abc"
    assert registry.get("ses_opencode_internal") is None


def test_client_session_id_honored_without_registered_api_key() -> None:
    registry = SessionRegistry()

    resolved = resolve_session_id(
        registry, {"Authorization": "Bearer not-a-session", "x-session-id": "client-1"}, {}
    )

    assert resolved == "client-1"
    assert registry.get("client-1") is not None


def test_body_session_id_does_not_override_registered_api_key() -> None:
    registry = SessionRegistry()
    registry.register("sk-polar-xyz", registered=True)

    resolved = resolve_session_id(
        registry, {"x-api-key": "sk-polar-xyz"}, {"_proxy_session_id": "other"}
    )

    assert resolved == "sk-polar-xyz"
