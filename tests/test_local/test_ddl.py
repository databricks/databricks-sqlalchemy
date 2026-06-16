import pytest
from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    Numeric,
    Uuid,
    create_engine,
    insert,
)
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

    def test_backtick_combined_with_default_escape_chars(self):
        """Column name with BOTH a literal backtick AND a character in
        SQLAlchemy's default ``bindname_escape_characters`` (``.``,
        ``[``, ``:``, ``%``, ...). The backtick path bypasses super's
        translation entirely so both characters render verbatim inside
        the backtick-quoted marker, and the params dict key stays the
        single-backtick, un-translated original.
        """
        metadata = MetaData()
        table = Table("t", metadata, Column("col`x.y", String()))
        compiled = self._compile_insert(table, {"col`x.y": "v"})
        sql = str(compiled)
        # Backtick doubled, dot preserved
        assert ":`col``x.y`" in sql
        params = compiled.construct_params()
        assert params["col`x.y"] == "v"
        # No mapping side effects
        assert compiled.escaped_bind_names == {}

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
        table = Table(
            "t", metadata, Column("select", String()), Column("from", String())
        )
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
        expanded = compiled.construct_expanded_state({"col-name_1": ["a", "b", "c"]})
        assert ":`col-name_1_1`" in expanded.statement
        assert ":`col-name_1_2`" in expanded.statement
        assert ":`col-name_1_3`" in expanded.statement


class TestMultiRowInsertCasts(DDLTestBase):
    def test_multi_values_casts_mixed_type_column(self):
        metadata = MetaData()
        table = Table(
            "t", metadata, Column("name", String()), Column("value", String())
        )
        stmt = insert(table).values(
            [
                {"name": "alice", "value": 1},
                {"name": "bob", "value": 0},
                {"name": None, "value": "NE"},
            ]
        )

        sql = str(stmt.compile(bind=self.engine))

        assert "CAST(:`value_m0` AS STRING)" in sql
        assert "CAST(:`value_m1` AS STRING)" in sql
        assert "CAST(:`value_m2` AS STRING)" in sql
        assert "CAST(:`name_m0` AS STRING)" not in sql
        assert "CAST(:`name_m1` AS STRING)" not in sql
        assert "CAST(:`name_m2` AS STRING)" not in sql

    def test_mixed_scalars_are_not_cast_for_non_string_targets(self):
        metadata = MetaData()
        table = Table("t", metadata, Column("value", Numeric()))
        stmt = insert(table).values([{"value": 1}, {"value": "not-a-number"}])

        sql = str(stmt.compile(bind=self.engine))
        assert "CAST(:`value_m0` AS DECIMAL)" not in sql
        assert "CAST(:`value_m1` AS DECIMAL)" not in sql

    def test_mixed_scalars_are_cast_for_string_compiled_types(self):
        metadata = MetaData()
        table = Table("t", metadata, Column("value", Uuid()))
        stmt = insert(table).values(
            [{"value": "00000000-0000-0000-0000-0000000000ff"}, {"value": 1}]
        )

        sql = str(stmt.compile(bind=self.engine))
        assert "CAST(:`value_m0` AS STRING)" in sql
        assert "CAST(:`value_m1` AS STRING)" in sql

    def test_bool_number_mixed_string_target_is_cast(self):
        metadata = MetaData()
        table = Table("t", metadata, Column("value", String()))
        stmt = insert(table).values([{"value": True}, {"value": 1}])

        sql = str(stmt.compile(bind=self.engine))
        assert "CAST(:`value_m0` AS STRING)" in sql
        assert "CAST(:`value_m1` AS STRING)" in sql

    def test_multi_value_casts_can_be_disabled_by_url_param(self):
        engine = create_engine(
            "databricks://token:****@****"
            "?http_path=****&catalog=****&schema=****"
            "&enable_multirow_insert_casts=false"
        )
        metadata = MetaData()
        table = Table("t", metadata, Column("value", String()))
        stmt = insert(table).values([{"value": 1}, {"value": 0}, {"value": "NE"}])

        sql = str(stmt.compile(bind=engine))
        assert "CAST(:`value_m0` AS STRING)" not in sql
        assert ":`value_m0`" in sql
        assert ":`value_m1`" in sql
        assert ":`value_m2`" in sql

    def test_empty_or_unknown_cast_gate_url_param_uses_default_enabled(self):
        for param_value in ("", "garbage", "flase"):
            engine = create_engine(
                "databricks://token:****@****"
                "?http_path=****&catalog=****&schema=****"
                f"&enable_multirow_insert_casts={param_value}"
            )
            metadata = MetaData()
            table = Table("t", metadata, Column("value", String()))
            stmt = insert(table).values([{"value": 1}, {"value": "NE"}])

            sql = str(stmt.compile(bind=engine))
            assert "CAST(:`value_m0` AS STRING)" in sql
            assert "CAST(:`value_m1` AS STRING)" in sql

    def test_homogeneous_multi_values_are_not_cast(self):
        metadata = MetaData()
        table = Table("t", metadata, Column("value", String()))
        stmt = insert(table).values([{"value": "A"}, {"value": "B"}, {"value": "C"}])

        sql = str(stmt.compile(bind=self.engine))
        assert "CAST(:`value_m0` AS STRING)" not in sql
        assert "CAST(:`value_m1` AS STRING)" not in sql
        assert "CAST(:`value_m2` AS STRING)" not in sql

    def test_numeric_family_multi_values_are_not_cast(self):
        metadata = MetaData()
        table = Table("t", metadata, Column("score", Numeric()))
        stmt = insert(table).values([{"score": 1}, {"score": 2.5}, {"score": 3}])

        sql = str(stmt.compile(bind=self.engine))
        assert "CAST(:`score_m0` AS DECIMAL)" not in sql
        assert "CAST(:`score_m1` AS DECIMAL)" not in sql
        assert "CAST(:`score_m2` AS DECIMAL)" not in sql

    def test_single_row_insert_does_not_render_casts(self):
        metadata = MetaData()
        table = Table("t", metadata, Column("value", String()))
        stmt = insert(table).values({"value": "A"})

        sql = str(stmt.compile(bind=self.engine))
        assert "CAST(:`value` AS STRING)" not in sql


class TestMultiRowInsertCastsEscapedBindNames(DDLTestBase):
    """Regression tests for PECOBLR-2746 follow-up.

    SQLAlchemy's ``bindname_escape_characters`` translates the chars space,
    ``.``, ``[``, ``]``, ``(``, ``)``, ``%``, ``:`` in bind names to ``_`` etc.
    The cast pass keys off ``self.binds`` (raw names) but the rendered SQL uses
    the escaped form. If the cast pass doesn't look up the escaped name, the
    str.replace becomes a no-op and the mixed-type insert fails again. These
    tests pin the fix.
    """

    # Each pair is (column_name, expected_bind_token_inside_backticks).
    # The expected token mirrors SQLAlchemy's default bindname_escape_characters
    # map: space/./[/] → _, ( → A, ) → Z, % → P, : → C.
    _ESCAPED_NAMES = [
        ("col with space", "col_with_space"),
        ("col.dot", "col_dot"),
        ("col[bracket]", "col_bracket_"),
        ("col(paren)", "colAparenZ"),
        ("col%pct", "colPpct"),
        ("col:colon", "colCcolon"),
    ]

    @pytest.mark.parametrize("column_name,escaped_token", _ESCAPED_NAMES)
    def test_cast_renders_for_escape_char_column(self, column_name, escaped_token):
        metadata = MetaData()
        table = Table("t", metadata, Column(column_name, String()))
        stmt = insert(table).values(
            [{column_name: 1}, {column_name: 0}, {column_name: "NE"}]
        )

        sql = str(stmt.compile(bind=self.engine))

        for idx in range(3):
            marker = f":`{escaped_token}_m{idx}`"
            cast = f"CAST({marker} AS STRING)"
            assert cast in sql, (
                f"expected {cast!r} in compiled SQL for column "
                f"{column_name!r}, got:\n{sql}"
            )
            # And the bare marker must not appear standalone — every
            # occurrence of it must be inside the CAST(...).
            assert sql.count(marker) == sql.count(cast), (
                f"bare {marker!r} appears outside CAST(...) for column "
                f"{column_name!r}:\n{sql}"
            )

    def test_cast_renders_for_backtick_column_name(self):
        """Literal-backtick column: bindparam_string short-circuits super and
        doubles the backtick directly, so escaped_bind_names stays empty. Our
        `.get(bind_name, bind_name)` falls back to the raw name and the
        `.replace("`", "``")` in the marker rebuild reproduces the same
        doubling, so the marker matches and CAST wraps correctly.
        """
        metadata = MetaData()
        table = Table("t", metadata, Column("col`tick", String()))
        stmt = insert(table).values(
            [{"col`tick": 1}, {"col`tick": 0}, {"col`tick": "NE"}]
        )

        sql = str(stmt.compile(bind=self.engine))
        for idx in range(3):
            assert f"CAST(:`col``tick_m{idx}` AS STRING)" in sql, sql

    def test_cast_renders_for_backtick_plus_escape_char(self):
        """Both backtick and a default-escape-map char in the same column name.
        The backtick path bypasses super entirely (so the escape map never
        runs), and `.replace("`", "``")` doubles the backtick — the dot stays
        verbatim inside the backtick-quoted marker.
        """
        metadata = MetaData()
        table = Table("t", metadata, Column("col`x.y", String()))
        stmt = insert(table).values([{"col`x.y": 1}, {"col`x.y": 0}, {"col`x.y": "NE"}])

        sql = str(stmt.compile(bind=self.engine))
        for idx in range(3):
            assert f"CAST(:`col``x.y_m{idx}` AS STRING)" in sql, sql

    def test_cast_renders_for_mixed_escape_chars_in_same_table(self):
        metadata = MetaData()
        table = Table(
            "t",
            metadata,
            Column("a b", String()),
            Column("c.d", String()),
            Column("e", String()),
        )
        stmt = insert(table).values(
            [
                {"a b": 1, "c.d": 1, "e": 1},
                {"a b": 0, "c.d": 0, "e": 0},
                {"a b": "NE", "c.d": "NE", "e": "NE"},
            ]
        )

        sql = str(stmt.compile(bind=self.engine))
        for token in ("a_b", "c_d", "e"):
            for idx in range(3):
                assert f"CAST(:`{token}_m{idx}` AS STRING)" in sql, sql
