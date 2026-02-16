"""Tests for DatabricksDialect.is_disconnect() method."""
import pytest
from databricks.sqlalchemy import DatabricksDialect
from databricks.sql.exc import (
    Error,
    InterfaceError,
    DatabaseError,
    OperationalError,
    RequestError,
    SessionAlreadyClosedError,
    CursorAlreadyClosedError,
    MaxRetryDurationError,
    NonRecoverableNetworkError,
    UnsafeToRetryError,
)


class TestIsDisconnect:
    @pytest.fixture
    def dialect(self):
        return DatabricksDialect()

    # --- InterfaceError: closed connection/cursor (client.py) ---

    def test_interface_error_closed_connection(self, dialect):
        """All InterfaceError messages with 'closed' are disconnects."""
        test_cases = [
            InterfaceError("Cannot create cursor from closed connection"),
            InterfaceError("Cannot get autocommit on closed connection"),
            InterfaceError("Cannot set autocommit on closed connection"),
            InterfaceError("Cannot commit on closed connection"),
            InterfaceError("Cannot rollback on closed connection"),
            InterfaceError("Cannot get transaction isolation on closed connection"),
            InterfaceError("Cannot set transaction isolation on closed connection"),
            InterfaceError("Attempting operation on closed cursor"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is True

    def test_interface_error_without_closed_not_disconnect(self, dialect):
        """InterfaceError without 'closed' is not a disconnect."""
        error = InterfaceError("Some other interface error")
        assert dialect.is_disconnect(error, None, None) is False

    # --- RequestError: transport/network-level errors ---

    def test_request_error_is_disconnect(self, dialect):
        """All RequestError instances are disconnects."""
        test_cases = [
            RequestError("HTTP client is closing or has been closed"),
            RequestError("Connection pool not initialized"),
            RequestError("HTTP request failed: max retries exceeded"),
            RequestError("HTTP request error: connection reset"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is True

    def test_request_error_subclasses_are_disconnect(self, dialect):
        """RequestError subclasses are all disconnects."""
        test_cases = [
            SessionAlreadyClosedError("Session already closed"),
            CursorAlreadyClosedError("Cursor already closed"),
            MaxRetryDurationError("Retry duration exceeded"),
            NonRecoverableNetworkError("HTTP 501"),
            UnsafeToRetryError("Unexpected HTTP error"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is True

    # --- DatabaseError: server-side session/operation errors ---

    def test_database_error_with_invalid_handle(self, dialect):
        """DatabaseError with 'invalid handle' is a disconnect."""
        test_cases = [
            DatabaseError("Invalid SessionHandle"),
            DatabaseError("[Errno INVALID_HANDLE] Session does not exist"),
            DatabaseError("INVALID HANDLE"),
            DatabaseError("invalid handle"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is True

    def test_database_error_unexpectedly_closed_server_side(self, dialect):
        """DatabaseError for operations closed server-side is a disconnect."""
        test_cases = [
            DatabaseError("Command abc123 unexpectedly closed server side"),
            DatabaseError("Command None unexpectedly closed server side"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is True

    def test_database_error_without_disconnect_indicators(self, dialect):
        """DatabaseError without disconnect indicators is not a disconnect."""
        test_cases = [
            DatabaseError("Syntax error in SQL"),
            DatabaseError("Table not found"),
            DatabaseError("Permission denied"),
            DatabaseError("Catalog name is required for get_schemas"),
            DatabaseError("Catalog name is required for get_columns"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is False

    # --- OperationalError (non-RequestError) ---

    def test_operational_error_not_disconnect(self, dialect):
        """OperationalError without disconnect indicators is not a disconnect."""
        test_cases = [
            OperationalError("Timeout waiting for query"),
            OperationalError("Empty TColumn instance"),
            OperationalError("Unsupported TRowSet instance"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is False

    # --- Base Error class: older connector versions (client.py:385) ---

    def test_base_error_closed_connection_is_disconnect(self, dialect):
        """Base Error with 'closed connection/cursor' is a disconnect.

        Older released versions of databricks-sql-connector raise Error
        (not InterfaceError) for closed connection messages.
        """
        test_cases = [
            Error("Cannot create cursor from closed connection"),
            Error("Cannot get autocommit on closed connection"),
            Error("Attempting operation on closed cursor"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is True

    def test_base_error_without_closed_not_disconnect(self, dialect):
        """Base Error without 'closed connection/cursor' is not a disconnect."""
        test_cases = [
            Error("Some other error"),
            Error("Connection timeout"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is False

    # --- Other exceptions ---

    def test_other_errors_not_disconnect(self, dialect):
        """Non-connector exception types are not disconnects."""
        test_cases = [
            Exception("Some random error"),
            ValueError("Bad value"),
            RuntimeError("Runtime failure"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is False
