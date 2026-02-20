"""Tests for DatabricksDialect.do_ping() method."""
from unittest.mock import MagicMock, patch
import pytest
from databricks.sqlalchemy import DatabricksDialect


class TestDoPing:
    @pytest.fixture
    def dialect(self):
        return DatabricksDialect()

    def test_ping_success(self, dialect):
        """do_ping returns True when SELECT 1 succeeds."""
        mock_conn = MagicMock()
        assert dialect.do_ping(mock_conn) is True
        mock_conn.cursor.assert_called_once()
        mock_conn.cursor().execute.assert_called_once_with("SELECT 1")
        mock_conn.cursor().close.assert_called_once()

    def test_ping_cursor_fails(self, dialect):
        """do_ping returns False when cursor() raises (connection closed)."""
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("Cannot create cursor from closed connection")
        assert dialect.do_ping(mock_conn) is False

    def test_ping_execute_fails(self, dialect):
        """do_ping returns False when execute() raises (session expired)."""
        mock_conn = MagicMock()
        mock_conn.cursor().execute.side_effect = Exception("Invalid SessionHandle")
        assert dialect.do_ping(mock_conn) is False

    def test_ping_cursor_closed_on_success(self, dialect):
        """Cursor is closed after a successful ping."""
        mock_conn = MagicMock()
        dialect.do_ping(mock_conn)
        mock_conn.cursor().close.assert_called_once()

    def test_ping_cursor_closed_on_execute_failure(self, dialect):
        """Cursor is closed even when execute() fails."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("network error")
        dialect.do_ping(mock_conn)
        mock_cursor.close.assert_called_once()
