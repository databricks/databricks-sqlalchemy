from unittest.mock import patch

from databricks.sqlalchemy import DatabricksDialect


class TestReflectionCache:
    """Reflection methods backed by DESCRIBE TABLE EXTENDED must honour info_cache.

    SQLAlchemy passes an ``info_cache`` dict through every ``Inspector`` call so
    that one reflection pass issues one round-trip per table. A dialect method
    only participates when it is decorated with ``@reflection.cache``.
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
