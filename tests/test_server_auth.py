"""H7: the demo server binds localhost, requires a bearer token, ships no default secret."""
from __future__ import annotations

import pytest

from config import Config, ServerConfig


def TestClient(app):  # noqa: N802 — fastapi/httpx are optional server extras
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient as _TC
    return _TC(app)


def _server(tmp_path, token="s3cret-token"):
    pytest.importorskip("fastapi")
    from server import SpectRAGServer
    cfg = Config(store_path=tmp_path / "idx")
    cfg.server.api_token = token
    return SpectRAGServer(cfg)


ROUTES = [("GET", "/api/status"), ("GET", "/api/sessions"), ("POST", "/api/sessions"),
          ("GET", "/api/sessions/x"), ("DELETE", "/api/sessions/x"),
          ("GET", "/api/sessions/x/messages"), ("GET", "/api/sessions/x/graph"),
          ("POST", "/api/sessions/x/query")]


@pytest.mark.parametrize("method,path", ROUTES)
def test_every_route_rejects_missing_or_wrong_token(tmp_path, method, path):
    client = TestClient(_server(tmp_path).app)
    assert client.request(method, path).status_code == 401
    assert client.request(method, path,
                          headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.request(method, path,
                          headers={"Authorization": "Basic s3cret-token"}).status_code == 401


def test_valid_token_is_accepted(tmp_path):
    client = TestClient(_server(tmp_path).app)
    r = client.get("/api/sessions", headers={"Authorization": "Bearer s3cret-token"})
    assert r.status_code == 200 and r.json() == []


def test_default_bind_is_localhost_and_no_default_token():
    assert ServerConfig().host == "127.0.0.1"
    assert ServerConfig().api_token is None


def test_unset_token_generates_a_random_one_per_server(tmp_path, monkeypatch):
    monkeypatch.delenv("SPECTRAG_SERVER_TOKEN", raising=False)
    a, b = _server(tmp_path / "a", None), _server(tmp_path / "b", None)
    assert a.api_token and len(a.api_token) >= 24 and a.api_token != b.api_token
    client = TestClient(a.app)
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions",
                      headers={"Authorization": f"Bearer {a.api_token}"}).status_code == 200


def test_token_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRAG_SERVER_TOKEN", "from-env")
    s = _server(tmp_path, None)
    assert s.api_token == "from-env"
