import pytest
from sqlalchemy import Column, MetaData, String, Table, Numeric, Integer, create_engine
from sqlalchemy.schema import (
    CreateTable,
    DropColumnComment,
    DropTableComment,
    SetColumnComment,
    SetTableComment,
)
from databricks.sqlalchemy import DatabricksArray, DatabricksMap


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
        table = Table("complex_type", metadata, col1, col2)
        return metadata

    def test_create_table_with_complex_type(self, metadata):
        stmt = CreateTable(metadata.tables["complex_type"])
        output = self.compile(stmt)

        assert "array_array_string ARRAY<ARRAY<STRING>>" in output
        assert "map_string_string MAP<STRING,STRING>" in output
