import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

import requests
from databricks.sql.auth.authenticators import CredentialsProvider
from databricks.sql.auth.endpoint import get_oauth_endpoints


class ServicePrincipalConfigurationError(ValueError):
    """Raised when the service principal configuration is incomplete."""


class ServicePrincipalAuthenticationError(RuntimeError):
    """Raised when fetching an OAuth token fails."""


def _normalize_hostname(hostname: str) -> str:
    maybe_scheme = "" if hostname.startswith("https://") else "https://"
    trimmed = (
        hostname[len("https://") :] if hostname.startswith("https://") else hostname
    )
    return f"{maybe_scheme}{trimmed}".rstrip("/")


class ServicePrincipalCredentialsProvider(CredentialsProvider):
    """CredentialsProvider that performs the Databricks OAuth client credentials flow."""

    DEFAULT_SCOPES = ("sql",)

    def __init__(
        self,
        server_hostname: str,
        client_id: str,
        client_secret: str,
        *,
        scopes: Optional[Iterable[str]] = None,
        refresh_margin: int = 60,
        request_timeout: int = 10,
    ):
        if not server_hostname:
            raise ServicePrincipalConfigurationError("server_hostname is required")
        if not client_id:
            raise ServicePrincipalConfigurationError("client_id is required")
        if not client_secret:
            raise ServicePrincipalConfigurationError("client_secret is required")

        self._hostname = _normalize_hostname(server_hostname)
        oauth_endpoints = get_oauth_endpoints(self._hostname, use_azure_auth=False)
        if not oauth_endpoints:
            raise ServicePrincipalConfigurationError(
                f"Unable to determine OAuth endpoints for host {server_hostname}"
            )

        scope_tuple = tuple(scopes) if scopes else self.DEFAULT_SCOPES
        mapped_scopes = oauth_endpoints.get_scopes_mapping(list(scope_tuple))

        self._client_id = client_id
        self._client_secret = client_secret
        self._scopes: List[str] = mapped_scopes
        self._refresh_margin = refresh_margin
        self._request_timeout = request_timeout
        self._access_token: Optional[str] = None
        self._expires_at: float = 0
        self._lock = threading.Lock()
        self._token_endpoint = self._discover_token_endpoint(oauth_endpoints)

    def auth_type(self) -> str:
        return "databricks-service-principal"

    def __call__(self) -> Callable[[], Dict[str, str]]:
        def header_factory() -> Dict[str, str]:
            access_token = self._get_token()
            return {"Authorization": f"Bearer {access_token}"}

        return header_factory

    def _discover_token_endpoint(self, oauth_endpoints) -> str:
        openid_config_url = oauth_endpoints.get_openid_config_url(self._hostname)
        try:
            response = requests.get(openid_config_url, timeout=self._request_timeout)
            response.raise_for_status()
            config = response.json()
        except Exception as exc:
            raise ServicePrincipalAuthenticationError(
                "Failed to load Databricks OAuth configuration"
            ) from exc

        token_endpoint = config.get("token_endpoint")
        if not token_endpoint:
            raise ServicePrincipalAuthenticationError(
                "OAuth configuration did not include a token endpoint"
            )
        return token_endpoint

    def _needs_refresh(self) -> bool:
        if not self._access_token:
            return True
        now = time.time()
        return now >= (self._expires_at - self._refresh_margin)

    def _get_token(self) -> str:
        with self._lock:
            if self._needs_refresh():
                self._refresh_token()
            assert self._access_token
            return self._access_token

    def _refresh_token(self) -> None:

        payload = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "scope": " ".join(self._scopes),
        }

        response = requests.post(
            self._token_endpoint, data=payload, timeout=self._request_timeout
        )
        try:
            response.raise_for_status()
        except Exception as exc:
            raise ServicePrincipalAuthenticationError(
                "Failed to retrieve OAuth token for service principal"
            ) from exc

        try:
            parsed = response.json()
            access_token = parsed["access_token"]
        except Exception as exc:  # pragma: no cover - defensive
            raise ServicePrincipalAuthenticationError(
                "OAuth response did not include an access token"
            ) from exc

        expires_in_raw = parsed.get("expires_in", 3600)
        try:
            expires_in = int(expires_in_raw)
        except (TypeError, ValueError):
            expires_in = 3600

        self._access_token = access_token
        self._expires_at = time.time() + max(expires_in, self._refresh_margin + 1)
