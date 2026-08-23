from contextlib import contextmanager
from unittest.mock import Mock, patch

from databricks.sqlalchemy import DatabricksDialect


@contextmanager
def _counting_cursor(self, connection):
    """Stand in for get_connection_cursor, recording each columns() call."""

    cursor = Mock()

    def columns(**kwargs):
        _counting_cursor.calls.append(kwargs["table_name"])
        result = Mock()
        result.fetchall.return_value = []
        return result

    cursor.columns.side_effect = columns
    yield cursor


class TestReflectionCache:
    """Reflection methods must honour the info_cache SQLAlchemy threads through.

    SQLAlchemy passes an ``info_cache`` dict through every ``Inspector`` call so
    that reflecting the same table repeatedly costs one round-trip rather than
    one per call. A dialect method only participates when it is decorated with
    ``@reflection.cache``.
    """

    def _dialect_and_cache(self):
        return DatabricksDialect(), {}

    def test_get_foreign_keys_is_cached(self):
        dialect, info_cache = self._dialect_and_cache()

        with patch.object(
            dialect, "_describe_table_extended", return_value=[]
        ) as mock_dte:
            first = dialect.get_foreign_keys(
                None, "some_table", schema="some_schema", info_cache=info_cache
            )
            second = dialect.get_foreign_keys(
                None, "some_table", schema="some_schema", info_cache=info_cache
            )

        assert mock_dte.call_count == 1
        assert first == second

    def test_get_foreign_keys_cache_is_per_table(self):
        dialect, info_cache = self._dialect_and_cache()

        with patch.object(
            dialect, "_describe_table_extended", return_value=[]
        ) as mock_dte:
            dialect.get_foreign_keys(
                None, "table_one", schema="some_schema", info_cache=info_cache
            )
            dialect.get_foreign_keys(
                None, "table_two", schema="some_schema", info_cache=info_cache
            )

        assert mock_dte.call_count == 2

    def test_get_foreign_keys_without_info_cache_is_not_cached(self):
        """Direct calls that pass no info_cache keep their existing behaviour."""

        dialect, _ = self._dialect_and_cache()

        with patch.object(
            dialect, "_describe_table_extended", return_value=[]
        ) as mock_dte:
            dialect.get_foreign_keys(None, "some_table", schema="some_schema")
            dialect.get_foreign_keys(None, "some_table", schema="some_schema")

        assert mock_dte.call_count == 2

    def _get_columns_calls(self, dialect, info_cache, times, table_name="some_table"):
        """Call get_columns `times` times, returning the server round-trip counts."""

        _counting_cursor.calls = []

        with patch.object(
            DatabricksDialect, "get_connection_cursor", _counting_cursor
        ), patch.object(
            dialect, "_describe_table_extended", return_value=[]
        ) as mock_dte:
            for _ in range(times):
                kwargs = {} if info_cache is None else {"info_cache": info_cache}
                dialect.get_columns(None, table_name, None, **kwargs)

        return len(_counting_cursor.calls), mock_dte.call_count

    def test_get_columns_is_cached(self):
        dialect, info_cache = self._dialect_and_cache()
        dialect.catalog = "some_catalog"
        dialect.schema = "some_schema"

        column_calls, describe_calls = self._get_columns_calls(
            dialect, info_cache, times=3
        )

        assert column_calls == 1
        # An empty columns() result makes get_columns fall back to
        # DESCRIBE TABLE EXTENDED to tell a column-less table from a missing
        # one, so an uncached call costs two round-trips, not one.
        assert describe_calls == 1

    def test_get_columns_without_info_cache_is_not_cached(self):
        """Direct calls that pass no info_cache keep their existing behaviour."""

        dialect, _ = self._dialect_and_cache()
        dialect.catalog = "some_catalog"
        dialect.schema = "some_schema"

        column_calls, describe_calls = self._get_columns_calls(dialect, None, times=3)

        assert column_calls == 3
        assert describe_calls == 3

    def test_get_pk_constraint_is_cached(self):
        """Guards the already-correct sibling against regression."""

        dialect, info_cache = self._dialect_and_cache()

        with patch.object(
            dialect, "_describe_table_extended", return_value=[]
        ) as mock_dte:
            dialect.get_pk_constraint(
                None, "some_table", schema="some_schema", info_cache=info_cache
            )
            dialect.get_pk_constraint(
                None, "some_table", schema="some_schema", info_cache=info_cache
            )

        assert mock_dte.call_count == 1
