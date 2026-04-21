import pytest
from sqlalchemy import Column, MetaData, String, Table, Numeric, Integer, create_engine, insert
from sqlalchemy.schema import (
    CreateTable,
    DropColumnComment,
    DropTableComment,
    SetColumnComment,
    SetTableComment,
)
from databricks.sqlalchemy import DatabricksArray, DatabricksMap, DatabricksVariant


class DDLTestBase:
    engine = create_engine(
        "databricks://token:****@****?http_path=****&catalog=****&schema=****"
    )

    def compile(self, stmt):
        return str(stmt.compile(bind=self.engine))


class TestColumnCommentDDL(DDLTestBase):
    @pytest.fixture
    def metadata(self) -> MetaData:
        """Assemble a metadata object with one table containing one column."""
        metadata = MetaData()

        column = Column("foo", String, comment="bar")
        table = Table("foobar", metadata, column)

        return metadata

    @pytest.fixture
    def table(self, metadata) -> Table:
        return metadata.tables.get("foobar")

    @pytest.fixture
    def column(self, table) -> Column:
        return table.columns[0]

    def test_create_table_with_column_comment(self, table):
        stmt = CreateTable(table)
        output = self.compile(stmt)

        # output is a CREATE TABLE statement
        assert "foo STRING COMMENT 'bar'" in output

    def test_alter_table_add_column_comment(self, column):
        stmt = SetColumnComment(column)
        output = self.compile(stmt)
        assert output == "ALTER TABLE foobar ALTER COLUMN foo COMMENT 'bar'"

    def test_alter_table_drop_column_comment(self, column):
        stmt = DropColumnComment(column)
        output = self.compile(stmt)
        assert output == "ALTER TABLE foobar ALTER COLUMN foo COMMENT ''"


class TestTableCommentDDL(DDLTestBase):
    @pytest.fixture
    def metadata(self) -> MetaData:
        """Assemble a metadata object with one table containing one column."""
        metadata = MetaData()

        col1 = Column("foo", String)
        col2 = Column("foo", String)
        tbl_w_comment = Table("martin", metadata, col1, comment="foobar")
        tbl_wo_comment = Table("prs", metadata, col2)

        return metadata

    @pytest.fixture
    def table_with_comment(self, metadata) -> Table:
        return metadata.tables.get("martin")

    @pytest.fixture
    def table_without_comment(self, metadata) -> Table:
        return metadata.tables.get("prs")

    def test_create_table_with_comment(self, table_with_comment):
        stmt = CreateTable(table_with_comment)
        output = self.compile(stmt)
        assert "USING DELTA" in output
        assert "COMMENT 'foobar'" in output

    def test_alter_table_add_comment(self, table_without_comment: Table):
        table_without_comment.comment = "wireless mechanical keyboard"
        stmt = SetTableComment(table_without_comment)
        output = self.compile(stmt)

        assert output == "COMMENT ON TABLE prs IS 'wireless mechanical keyboard'"

    def test_alter_table_drop_comment(self, table_with_comment):
        """The syntax for COMMENT ON is here: https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-comment.html"""
        stmt = DropTableComment(table_with_comment)
        output = self.compile(stmt)
        assert output == "COMMENT ON TABLE martin IS NULL"


class TestTableComplexTypeDDL(DDLTestBase):
    @pytest.fixture(scope="class")
    def metadata(self) -> MetaData:
        metadata = MetaData()
        col1 = Column("array_array_string", DatabricksArray(DatabricksArray(String)))
        col2 = Column("map_string_string", DatabricksMap(String, String))
        col3 = Column("variant_col", DatabricksVariant())
        table = Table("complex_type", metadata, col1, col2, col3)
        return metadata

    def test_create_table_with_complex_type(self, metadata):
        stmt = CreateTable(metadata.tables["complex_type"])
        output = self.compile(stmt)

        assert "array_array_string ARRAY<ARRAY<STRING>>" in output
        assert "map_string_string MAP<STRING,STRING>" in output
        assert "variant_col VARIANT" in output


class TestBindParamQuoting(DDLTestBase):
    """Regression tests for column names that contain characters which are not
    legal inside a bare Databricks named-parameter marker (`:name`). Without
    the custom ``bindparam_string`` override, a column like
    ``col-with-hyphen`` produces SQL like ``VALUES (:col-with-hyphen)`` which
    fails with UNBOUND_SQL_PARAMETER on the server. The fix wraps such names
    in backticks (``VALUES (:`col-with-hyphen`)``), which the Databricks SQL
    grammar accepts as a quoted parameter identifier.
    """

    def _compile_insert(self, table, values):
        stmt = insert(table).values(values)
        return stmt.compile(bind=self.engine)

    def test_hyphenated_column_renders_backticked_bind_marker(self):
        metadata = MetaData()
        table = Table(
            "t",
            metadata,
            Column("col-with-hyphen", String()),
            Column("normal_col", String()),
        )
        compiled = self._compile_insert(
            table, {"col-with-hyphen": "x", "normal_col": "y"}
        )

        sql = str(compiled)
        # Hyphenated name is wrapped in backticks at the marker site
        assert ":`col-with-hyphen`" in sql
        # Plain name is untouched
        assert ":normal_col" in sql
        # The params dict sent to the driver keeps the ORIGINAL unquoted key
        # — this matches what the Databricks server expects (verified
        # empirically: a backticked marker `:`name`` binds against a plain
        # `name` key in the params dict).
        params = compiled.construct_params()
        assert params["col-with-hyphen"] == "x"
        assert params["normal_col"] == "y"
        assert "`col-with-hyphen`" not in params

    def test_hyphen_and_underscore_columns_do_not_collide(self):
        """A table containing both ``col-name`` and ``col_name`` must produce
        two distinct bind parameters with two distinct dict keys; otherwise
        one value would silently clobber the other.
        """
        metadata = MetaData()
        table = Table(
            "t",
            metadata,
            Column("col-name", String()),
            Column("col_name", String()),
        )
        compiled = self._compile_insert(
            table, {"col-name": "hyphen_value", "col_name": "underscore_value"}
        )

        sql = str(compiled)
        assert ":`col-name`" in sql
        assert ":col_name" in sql

        params = compiled.construct_params()
        assert params["col-name"] == "hyphen_value"
        assert params["col_name"] == "underscore_value"

    def test_plain_identifier_bind_names_are_unchanged(self):
        """No regression: ordinary column names must not be backticked."""
        metadata = MetaData()
        table = Table(
            "t",
            metadata,
            Column("id", String()),
            Column("name", String()),
        )
        compiled = self._compile_insert(table, {"id": "1", "name": "n"})
        sql = str(compiled)
        assert ":id" in sql
        assert ":name" in sql
        assert ":`id`" not in sql
        assert ":`name`" not in sql

    def test_space_and_dot_in_column_name_also_backticked(self):
        """The bare-identifier check covers all non-[A-Za-z0-9_] characters,
        not just hyphens — spaces, dots, etc. should also be wrapped.
        """
        metadata = MetaData()
        table = Table(
            "t",
            metadata,
            Column("col with space", String()),
            Column("col.with.dot", String()),
        )
        compiled = self._compile_insert(
            table, {"col with space": "s", "col.with.dot": "d"}
        )
        sql = str(compiled)
        assert ":`col with space`" in sql
        assert ":`col.with.dot`" in sql

        params = compiled.construct_params()
        assert params["col with space"] == "s"
        assert params["col.with.dot"] == "d"

    def test_leading_digit_column_is_backticked(self):
        """Databricks bind names cannot start with a digit either."""
        metadata = MetaData()
        table = Table("t", metadata, Column("1col", String()))
        compiled = self._compile_insert(table, {"1col": "x"})
        sql = str(compiled)
        assert ":`1col`" in sql

        params = compiled.construct_params()
        assert params["1col"] == "x"
