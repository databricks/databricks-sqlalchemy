"""Unit tests for do_ping() method in DatabricksDialect."""
import pytest
from unittest.mock import Mock
from databricks.sqlalchemy import DatabricksDialect


class TestDoPing:
    """Test the do_ping() method for connection health checks."""

    @pytest.fixture
    def dialect(self):
        """Create a DatabricksDialect instance."""
        return DatabricksDialect()

    def test_do_ping_success(self, dialect):
        """Test do_ping returns True when connection is alive."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        result = dialect.do_ping(mock_connection)

        assert result is True
        mock_cursor.execute.assert_called_once_with("SELECT VERSION()")
        mock_cursor.fetchone.assert_called_once()
        mock_cursor.close.assert_called_once()

    def test_do_ping_failure_cursor_creation(self, dialect):
        """Test do_ping returns False when cursor creation fails."""
        mock_connection = Mock()
        mock_connection.cursor.side_effect = Exception("Connection closed")

        result = dialect.do_ping(mock_connection)

        assert result is False

    def test_do_ping_failure_execute_and_cursor_closes(self, dialect):
        """Test do_ping returns False on execute error and cursor is closed."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("Query failed")

        result = dialect.do_ping(mock_connection)

        assert result is False
        mock_cursor.close.assert_called_once()
