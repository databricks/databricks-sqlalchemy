from typing import Dict

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

import databricks.sqlalchemy._service_principal as sp
from databricks.sqlalchemy.base import DatabricksDialect
from databricks.sqlalchemy._service_principal import (
    ServicePrincipalConfigurationError,
    ServicePrincipalCredentialsProvider,
)


class DummyResponse:
    def __init__(self, payload: Dict[str, object], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if not 200 <= self.status_code < 300:
            raise RuntimeError("request failed")
        return None

    def json(self):
        return self._payload


def test_service_principal_provider_refreshes_tokens(monkeypatch):
    responses = iter(
        [
            {"access_token": "token-one", "expires_in": 5},
            {"access_token": "token-two", "expires_in": 5},
        ]
    )
    call_count = {"value": 0}
    token_endpoint = "https://dbc.cloud.databricks.com/oidc/oauth2/v2.0/token"

    def fake_get(url, timeout):
        assert "oidc" in url
        return DummyResponse({"token_endpoint": token_endpoint})

    def fake_post(url, data, timeout):
        assert url == token_endpoint
        call_count["value"] += 1
        assert data["client_id"] == "client-id"
        return DummyResponse(next(responses))

    current_time = {"value": 0}

    def fake_time():
        return current_time["value"]

    monkeypatch.setattr(sp.requests, "post", fake_post)
    monkeypatch.setattr(sp.requests, "get", fake_get)
    monkeypatch.setattr(sp.time, "time", fake_time)

    provider = sp.ServicePrincipalCredentialsProvider(
        "dbc.cloud.databricks.com",
        "client-id",
        "secret",
        refresh_margin=0,
    )
    header_factory = provider()
    assert header_factory()["Authorization"] == "Bearer token-one"
    assert call_count["value"] == 1

    current_time["value"] = 2
    assert header_factory()["Authorization"] == "Bearer token-one"
    assert call_count["value"] == 1

    current_time["value"] = 6
    assert header_factory()["Authorization"] == "Bearer token-two"
    assert call_count["value"] == 2


def test_service_principal_provider_requires_hostname():
    with pytest.raises(ServicePrincipalConfigurationError):
        sp.ServicePrincipalCredentialsProvider(
            server_hostname="",
            client_id="client",
            client_secret="secret",
        )


def test_create_connect_args_with_service_principal(monkeypatch):
    token_endpoint = "https://dbc.cloud.databricks.com/oidc/oauth2/v2.0/token"

    def fake_get(url, timeout):
        return DummyResponse({"token_endpoint": token_endpoint})

    def fake_post(url, data, timeout):
        return DummyResponse({"access_token": "token", "expires_in": 3600})

    monkeypatch.setattr(sp.requests, "get", fake_get)
    monkeypatch.setattr(sp.requests, "post", fake_post)

    dialect = DatabricksDialect()
    url = make_url(
        "databricks://client-id:client-secret@acme.cloud.databricks.com"
        "?http_path=/sql/1&catalog=main&schema=test"
        "&authentication=service_principal"
    )

    _, kwargs = dialect.create_connect_args(url)

    assert "access_token" not in kwargs
    assert isinstance(
        kwargs.get("credentials_provider"), ServicePrincipalCredentialsProvider
    )
    assert kwargs["server_hostname"] == "acme.cloud.databricks.com"


def test_create_connect_args_requires_secret():
    dialect = DatabricksDialect()
    url = make_url(
        "databricks://client-id@acme.cloud.databricks.com"
        "?http_path=/sql/1&catalog=main&schema=test&authentication=service_principal"
    )

    with pytest.raises(ArgumentError, match="client_secret"):
        dialect.create_connect_args(url)
