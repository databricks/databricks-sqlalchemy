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
    """Regression tests for bind-parameter quoting.

    Databricks named parameter markers (``:name``) must be bare identifiers
    (``[A-Za-z_][A-Za-z0-9_]*``) unless wrapped in backticks. Because
    DataFrame-origin column names frequently contain hyphens (a character
    that's legal inside a backtick-quoted column identifier but not in a
    bare bind marker), the dialect wraps every bind name in backticks
    unconditionally. The backticks are SQL-side quoting only — the params
    dict sent to the driver keeps the original unquoted key.

    The behavior is gated by ``DatabricksDialect.quote_bind_params`` which
    defaults to True; set ``?quote_bind_params=false`` in the URL to
    disable.
    """

    def _compile_insert(self, table, values, engine=None):
        stmt = insert(table).values(values)
        return stmt.compile(bind=engine or self.engine)

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
        # Both names are backticked at the marker site
        assert ":`col-with-hyphen`" in sql
        assert ":`normal_col`" in sql
        # The params dict sent to the driver keeps the ORIGINAL unquoted key
        # — this matches what the Databricks server expects (verified
        # empirically: a backticked marker ``:`name``` binds against a plain
        # ``name`` key in the params dict).
        params = compiled.construct_params()
        assert params["col-with-hyphen"] == "x"
        assert params["normal_col"] == "y"
        assert "`col-with-hyphen`" not in params
        assert "`normal_col`" not in params

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
        assert ":`col_name`" in sql

        params = compiled.construct_params()
        assert params["col-name"] == "hyphen_value"
        assert params["col_name"] == "underscore_value"

    def test_plain_identifier_bind_names_are_also_backticked(self):
        """Every bind name is wrapped unconditionally — the Databricks SQL
        grammar accepts ``:`id``` identically to ``:id`` for plain names
        (verified against a live warehouse).
        """
        metadata = MetaData()
        table = Table(
            "t",
            metadata,
            Column("id", String()),
            Column("name", String()),
        )
        compiled = self._compile_insert(table, {"id": "1", "name": "n"})
        sql = str(compiled)
        assert ":`id`" in sql
        assert ":`name`" in sql


    def test_leading_digit_column_is_backticked(self):
        """Databricks bind names cannot start with a digit bare."""
        metadata = MetaData()
        table = Table("t", metadata, Column("1col", String()))
        compiled = self._compile_insert(table, {"1col": "x"})
        assert ":`1col`" in str(compiled)

    def test_literal_backtick_in_column_name_is_doubled(self):
        """A literal backtick inside a column name must be doubled in the
        rendered SQL (both the DDL column identifier and the bind
        marker), per the Spark SQL ``BACKQUOTED_IDENTIFIER`` lexer rule.
        The params dict key stays the single-backtick original — the
        server un-doubles when it parses the marker name.
        """
        from sqlalchemy.schema import CreateTable

        metadata = MetaData()
        table = Table("t", metadata, Column("a`b", String()))

        create_sql = str(CreateTable(table).compile(bind=self.engine))
        assert "`a``b`" in create_sql  # DDL identifier doubled

        compiled = self._compile_insert(table, {"a`b": "x"})
        assert ":`a``b`" in str(compiled)  # bind marker doubled
        params = compiled.construct_params()
        assert params["a`b"] == "x"  # dict key stays single-backtick
        assert "a``b" not in params

    def test_many_special_characters_in_column_names(self):
        """Column names containing characters that Delta allows (hyphens,
        slashes, question marks, hash, plus, star, at, dollar, amp, pipe,
        lt/gt) should render as valid backtick-quoted bind markers. We
        intentionally exclude characters Delta rejects at DDL time
        (space, parens, comma, equals) — those never land in a real
        Databricks table, so never reach the bind-name path.
        """
        # Each of these survives a CREATE TABLE in Delta (verified empirically)
        # and appears verbatim inside the backtick-quoted bind name — the
        # default SQLAlchemy escape map does not translate any of them.
        pass_through = [
            "col-hyphen",
            "col/slash",
            "col?question",
            "col#hash",
            "col+plus",
            "col*star",
            "col@at",
            "col$dollar",
            "col&amp",
            "col|pipe",
            "col<lt>gt",
        ]
        metadata = MetaData()
        columns = [Column(n, String()) for n in pass_through]
        table = Table("t", metadata, *columns)
        values = {n: f"v-{i}" for i, n in enumerate(pass_through)}
        compiled = self._compile_insert(table, values)
        sql = str(compiled)
        params = compiled.construct_params()
        for n in pass_through:
            assert f":`{n}`" in sql, f"bind marker missing for {n!r}"
            assert params[n] == values[n]

    def test_chars_in_sqlalchemy_default_escape_map_still_work(self):
        """Characters already in SQLAlchemy's default
        ``bindname_escape_characters`` (``.``, ``[``, ``]``, ``:``, ``%``)
        are pre-translated by super's ``bindparam_string`` before our
        backtick template wraps the resulting name. The rendered bind
        name is the translated one (``col_with_dot``), inside backticks.
        ``construct_params`` uses ``escaped_bind_names`` to translate
        the customer's incoming dict key to match. Verified end-to-end
        against a live warehouse.
        """
        metadata = MetaData()
        table = Table(
            "t",
            metadata,
            Column("col.with.dot", String()),
            Column("col[bracket]", String()),
            Column("col:colon", String()),
            Column("col%percent", String()),
        )
        compiled = self._compile_insert(
            table,
            {
                "col.with.dot": "d",
                "col[bracket]": "b",
                "col:colon": "c",
                "col%percent": "p",
            },
        )
        sql = str(compiled)
        assert ":`col_with_dot`" in sql
        assert ":`col_bracket_`" in sql
        assert ":`colCcolon`" in sql
        assert ":`colPpercent`" in sql

        params = compiled.construct_params()
        assert params["col_with_dot"] == "d"
        assert params["colCcolon"] == "c"
        assert params["col_bracket_"] == "b"
        assert params["colPpercent"] == "p"

    def test_unicode_column_names(self):
        """Databricks allows arbitrary Unicode inside backtick-quoted
        identifiers. Bind parameter quoting must handle Unicode names too.
        """
        names = ["prénom", "姓名", "Straße"]
        metadata = MetaData()
        table = Table("t", metadata, *(Column(n, String()) for n in names))
        values = {n: f"v{i}" for i, n in enumerate(names)}
        compiled = self._compile_insert(table, values)
        sql = str(compiled)
        for n in names:
            assert f":`{n}`" in sql
        params = compiled.construct_params()
        for n in names:
            assert params[n] == values[n]

    def test_sql_reserved_word_as_column_name(self):
        """Reserved words used as column names must work as bind params too."""
        metadata = MetaData()
        table = Table("t", metadata, Column("select", String()), Column("from", String()))
        compiled = self._compile_insert(table, {"select": "s", "from": "f"})
        sql = str(compiled)
        assert ":`select`" in sql
        assert ":`from`" in sql

    def test_where_clause_with_hyphenated_column(self):
        """The quoting must also apply when the hyphenated column appears in
        a WHERE clause (SELECT / UPDATE / DELETE all share this path).
        """
        from sqlalchemy import select

        metadata = MetaData()
        table = Table("t", metadata, Column("col-name", String()))
        stmt = select(table).where(table.c["col-name"] == "x")
        compiled = stmt.compile(bind=self.engine)
        # SQLAlchemy anonymizes the bind as ``<column>_<n>`` — the hyphen
        # survives into the bind name, so it must still be backtick-quoted.
        assert ":`col-name_1`" in str(compiled)

    def test_multivalues_insert_disambiguates_with_backticked_markers(self):
        """Multi-row INSERT generates per-row suffixed bind names. Each
        suffixed name must still render backtick-quoted correctly.
        """
        metadata = MetaData()
        table = Table("t", metadata, Column("col-name", String()))
        stmt = insert(table).values([{"col-name": "a"}, {"col-name": "b"}])
        compiled = stmt.compile(bind=self.engine)
        sql = str(compiled)
        # SQLAlchemy emits e.g. `col-name_m0`, `col-name_m1` for row-level params
        assert ":`col-name_m0`" in sql
        assert ":`col-name_m1`" in sql

    def test_in_clause_with_hyphenated_column_compiles_to_postcompile(self):
        """The initial compilation leaves an IN clause as a POSTCOMPILE
        placeholder. The placeholder itself isn't a bind marker so no
        quoting is needed at this stage — the actual expanded markers
        (``:\\`col-name_1_1\\``, …) are rendered at expansion time by our
        ``_literal_execute_expanding_parameter`` override (see
        ``test_in_clause_expansion_renders_backticked_markers``).
        """
        from sqlalchemy import select

        metadata = MetaData()
        table = Table("t", metadata, Column("col-name", String()))
        stmt = select(table).where(table.c["col-name"].in_(["a", "b"]))
        sql = str(stmt.compile(bind=self.engine))
        assert "POSTCOMPILE_col-name_1" in sql

    def test_in_clause_expansion_renders_backticked_markers(self):
        """Exercise the three sites that invoke
        ``_literal_execute_expanding_parameter``:

        * normal execute-time expansion via ``construct_expanded_state``
        * ``compile_kwargs={'render_postcompile': True}`` — which fires
          inside super's ``__init__``, before any post-super subclass
          init would take effect
        """
        from sqlalchemy import select

        metadata = MetaData()
        table = Table("t", metadata, Column("col-name", String()))
        stmt = select(table).where(table.c["col-name"].in_(["a", "b", "c"]))

        # (1) render_postcompile=True at compile time — fires inside super __init__
        rendered = str(
            stmt.compile(bind=self.engine, compile_kwargs={"render_postcompile": True})
        )
        assert ":`col-name_1_1`" in rendered
        assert ":`col-name_1_2`" in rendered
        assert ":`col-name_1_3`" in rendered

        # (2) construct_expanded_state at execute time
        compiled = stmt.compile(bind=self.engine)
        expanded = compiled.construct_expanded_state(
            {"col-name_1": ["a", "b", "c"]}
        )
        assert ":`col-name_1_1`" in expanded.statement
        assert ":`col-name_1_2`" in expanded.statement
        assert ":`col-name_1_3`" in expanded.statement
