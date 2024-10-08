import pytest

class DatabricksImportError(Exception):
    pass

class TestLibraryDependencySuite:

    @pytest.mark.skipif(pytest.importorskip("databricks_sql_connector_core"), reason="databricks_sql_connector_core is present")
    def test_sql_core(self):
        with pytest.raises(DatabricksImportError, match="databricks_sql_connector_core module is not available"):
            try:
                import databricks
            except ImportError:
                raise DatabricksImportError("databricks_sql_connector_core module is not available")

    @pytest.mark.skipif(pytest.importorskip("sqlalchemy"), reason="SQLAlchemy is present")
    def test_sqlalchemy(self):
        with pytest.raises(DatabricksImportError, match="sqlalchemy module is not available"):
            try:
                import sqlalchemy
            except ImportError:
                raise DatabricksImportError("sqlalchemy module is not available")