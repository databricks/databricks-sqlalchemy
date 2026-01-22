"""Tests for DatabricksDialect.is_disconnect() method."""
import pytest
from databricks.sqlalchemy import DatabricksDialect
from databricks.sql.exc import InterfaceError, DatabaseError, OperationalError


class TestIsDisconnect:
    @pytest.fixture
    def dialect(self):
        return DatabricksDialect()

    def test_interface_error_is_disconnect(self, dialect):
        """InterfaceError (client-side) is always a disconnect."""
        error = InterfaceError("Cannot create cursor from closed connection")
        assert dialect.is_disconnect(error, None, None) is True

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

    def test_database_error_without_invalid_handle(self, dialect):
        """DatabaseError without 'invalid handle' is not a disconnect."""
        test_cases = [
            DatabaseError("Syntax error in SQL"),
            DatabaseError("Table not found"),
            DatabaseError("Permission denied"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is False

    def test_other_errors_not_disconnect(self, dialect):
        """Other exception types are not disconnects."""
        test_cases = [
            OperationalError("Timeout waiting for query"),
            Exception("Some random error"),
        ]
        for error in test_cases:
            assert dialect.is_disconnect(error, None, None) is False
