import enum
from uuid import UUID

import pytest
import sqlalchemy
from sqlalchemy import Column, MetaData, Table, select

from databricks.sqlalchemy.base import DatabricksDialect
from databricks.sqlalchemy._types import (
    DatabricksUUID,
    DatabricksVariant,
    TINYINT,
    TIMESTAMP,
    TIMESTAMP_NTZ,
)


class DatabricksDataType(enum.Enum):
    """https://docs.databricks.com/en/sql/language-manual/sql-ref-datatypes.html"""

    BIGINT = enum.auto()
    BINARY = enum.auto()
    BOOLEAN = enum.auto()
    DATE = enum.auto()
    DECIMAL = enum.auto()
    DOUBLE = enum.auto()
    FLOAT = enum.auto()
    INT = enum.auto()
    INTERVAL = enum.auto()
    VOID = enum.auto()
    SMALLINT = enum.auto()
    STRING = enum.auto()
    TIMESTAMP = enum.auto()
    TIMESTAMP_NTZ = enum.auto()
    TINYINT = enum.auto()
    ARRAY = enum.auto()
    MAP = enum.auto()
    STRUCT = enum.auto()
    VARIANT = enum.auto()


# Defines the way that SQLAlchemy CamelCase types are compiled into Databricks SQL types.
# Note: I wish I could define this within the TestCamelCaseTypesCompilation class, but pytest doesn't like that.
camel_case_type_map = {
    sqlalchemy.types.BigInteger: DatabricksDataType.BIGINT,
    sqlalchemy.types.LargeBinary: DatabricksDataType.BINARY,
    sqlalchemy.types.Boolean: DatabricksDataType.BOOLEAN,
    sqlalchemy.types.Date: DatabricksDataType.DATE,
    sqlalchemy.types.DateTime: DatabricksDataType.TIMESTAMP_NTZ,
    sqlalchemy.types.Double: DatabricksDataType.DOUBLE,
    sqlalchemy.types.Enum: DatabricksDataType.STRING,
    sqlalchemy.types.Float: DatabricksDataType.FLOAT,
    sqlalchemy.types.Integer: DatabricksDataType.INT,
    sqlalchemy.types.Interval: DatabricksDataType.TIMESTAMP_NTZ,
    sqlalchemy.types.Numeric: DatabricksDataType.DECIMAL,
    sqlalchemy.types.PickleType: DatabricksDataType.BINARY,
    sqlalchemy.types.SmallInteger: DatabricksDataType.SMALLINT,
    sqlalchemy.types.String: DatabricksDataType.STRING,
    sqlalchemy.types.Text: DatabricksDataType.STRING,
    sqlalchemy.types.Time: DatabricksDataType.STRING,
    sqlalchemy.types.Unicode: DatabricksDataType.STRING,
    sqlalchemy.types.UnicodeText: DatabricksDataType.STRING,
    sqlalchemy.types.Uuid: DatabricksDataType.STRING,
}


def dict_as_tuple_list(d: dict):
    """Return a list of [(key, value), ...] from a dictionary."""
    return [(key, value) for key, value in d.items()]


class CompilationTestBase:
    dialect = DatabricksDialect()

    def _assert_compiled_value(
        self, type_: sqlalchemy.types.TypeEngine, expected: DatabricksDataType
    ):
        """Assert that when type_ is compiled for the databricks dialect, it renders the DatabricksDataType name.

        This method initialises the type_ with no arguments.
        """
        compiled_result = type_().compile(dialect=self.dialect)  # type: ignore
        assert compiled_result == expected.name

    def _assert_compiled_value_explicit(
        self, type_: sqlalchemy.types.TypeEngine, expected: str
    ):
        """Assert that when type_ is compiled for the databricks dialect, it renders the expected string.

        This method expects an initialised type_ so that we can test how a TypeEngine created with arguments
        is compiled.
        """
        compiled_result = type_.compile(dialect=self.dialect)
        assert compiled_result == expected


class TestCamelCaseTypesCompilation(CompilationTestBase):
    """Per the sqlalchemy documentation[^1] here, the camel case members of sqlalchemy.types are
    are expected to work across all dialects. These tests verify that the types compile into valid
    Databricks SQL type strings. For example, the sqlalchemy.types.Integer() should compile as "INT".

    Truly custom types like STRUCT (notice the uppercase) are not expected to work across all dialects.
    We test these separately.

    Note that these tests have to do with type **name** compiliation. Which is separate from actually
    mapping values between Python and Databricks.

    Note: SchemaType and MatchType are not tested because it's not used in table definitions

    [1]: https://docs.sqlalchemy.org/en/20/core/type_basics.html#generic-camelcase-types
    """

    @pytest.mark.parametrize("type_, expected", dict_as_tuple_list(camel_case_type_map))
    def test_bare_camel_case_types_compile(self, type_, expected):
        self._assert_compiled_value(type_, expected)

    def test_numeric_renders_as_decimal_with_precision(self):
        self._assert_compiled_value_explicit(
            sqlalchemy.types.Numeric(10), "DECIMAL(10)"
        )

    def test_numeric_renders_as_decimal_with_precision_and_scale(self):
        self._assert_compiled_value_explicit(
            sqlalchemy.types.Numeric(10, 2), "DECIMAL(10, 2)"
        )


uppercase_type_map = {
    sqlalchemy.types.ARRAY: DatabricksDataType.ARRAY,
    sqlalchemy.types.BIGINT: DatabricksDataType.BIGINT,
    sqlalchemy.types.BINARY: DatabricksDataType.BINARY,
    sqlalchemy.types.BOOLEAN: DatabricksDataType.BOOLEAN,
    sqlalchemy.types.DATE: DatabricksDataType.DATE,
    sqlalchemy.types.DECIMAL: DatabricksDataType.DECIMAL,
    sqlalchemy.types.DOUBLE: DatabricksDataType.DOUBLE,
    sqlalchemy.types.FLOAT: DatabricksDataType.FLOAT,
    sqlalchemy.types.INT: DatabricksDataType.INT,
    sqlalchemy.types.SMALLINT: DatabricksDataType.SMALLINT,
    sqlalchemy.types.TIMESTAMP: DatabricksDataType.TIMESTAMP,
    TINYINT: DatabricksDataType.TINYINT,
    TIMESTAMP: DatabricksDataType.TIMESTAMP,
    TIMESTAMP_NTZ: DatabricksDataType.TIMESTAMP_NTZ,
    DatabricksVariant: DatabricksDataType.VARIANT,
}


class TestUppercaseTypesCompilation(CompilationTestBase):
    """Per the sqlalchemy documentation[^1], uppercase types are considered to be specific to some
    database backends. These tests verify that the types compile into valid Databricks SQL type strings.

    [1]: https://docs.sqlalchemy.org/en/20/core/type_basics.html#backend-specific-uppercase-datatypes
    """

    @pytest.mark.parametrize("type_, expected", dict_as_tuple_list(uppercase_type_map))
    def test_bare_uppercase_types_compile(self, type_, expected):
        if isinstance(type_, type(sqlalchemy.types.ARRAY)):
            # ARRAY cannot be initialised without passing an item definition so we test separately
            # I preserve it in the uppercase_type_map for clarity
            assert True
        else:
            self._assert_compiled_value(type_, expected)

    def test_array_string_renders_as_array_of_string(self):
        """SQLAlchemy's ARRAY type requires an item definition. And their docs indicate that they've only tested
        it with Postgres since that's the only first-class dialect with support for ARRAY.

        https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.ARRAY
        """
        self._assert_compiled_value_explicit(
            sqlalchemy.types.ARRAY(sqlalchemy.types.String), "ARRAY<STRING>"
        )


class TestDatabricksUUID:
    """Regression coverage for github.com/databricks/databricks-sqlalchemy/issues/50.

    SQLAlchemy's default Uuid renders the 32-char hex form (no dashes) on backends
    without a native UUID type, which breaks equality against UUIDs stored as
    canonical 8-4-4-4-12 strings in Databricks.
    """

    dialect = DatabricksDialect()
    HYPHENATED = "1daa91d7-8d35-4684-86d6-3fa89042c1f4"
    HEX = "1daa91d78d35468486d63fa89042c1f4"
    sample = UUID(HYPHENATED)

    def test_bind_processor_preserves_hyphenated_form(self):
        process = DatabricksUUID().bind_processor(self.dialect)
        assert process(self.sample) == self.HYPHENATED

    def test_bind_processor_handles_none(self):
        process = DatabricksUUID().bind_processor(self.dialect)
        assert process(None) is None

    def test_literal_processor_renders_hyphenated_form(self):
        process = DatabricksUUID().literal_processor(self.dialect)
        assert process(self.sample) == "'%s'" % self.HYPHENATED

    def test_literal_processor_handles_none(self):
        process = DatabricksUUID().literal_processor(self.dialect)
        assert process(None) == "NULL"

    def test_result_processor_accepts_both_forms(self):
        process = DatabricksUUID().result_processor(self.dialect, None)
        assert process(self.HYPHENATED) == self.sample
        assert process(self.HEX) == self.sample
        assert process(None) is None

    def test_dialect_routes_uuid_to_databricks_uuid(self):
        """The colspecs entry is what makes a plain ``Uuid`` column use our impl."""
        assert self.dialect.colspecs[sqlalchemy.types.Uuid] is DatabricksUUID

    def test_uuid_where_clause_renders_with_dashes(self):
        meta = MetaData()
        users = Table(
            "users", meta, Column("id", sqlalchemy.types.Uuid, primary_key=True)
        )
        stmt = select(users).where(users.c.id == self.sample)

        literal_sql = str(
            stmt.compile(dialect=self.dialect, compile_kwargs={"literal_binds": True})
        )
        assert "'%s'" % self.HYPHENATED in literal_sql
        assert self.HEX not in literal_sql

    def test_uuid_bound_param_wire_value_has_dashes(self):
        meta = MetaData()
        users = Table(
            "users", meta, Column("id", sqlalchemy.types.Uuid, primary_key=True)
        )
        stmt = select(users).where(users.c.id == self.sample)
        compiled = stmt.compile(dialect=self.dialect)

        raw = compiled.construct_params()
        processed = {
            key: (
                compiled._bind_processors[key](value)
                if key in compiled._bind_processors
                else value
            )
            for key, value in raw.items()
        }
        assert self.HYPHENATED in processed.values()

    def test_bind_processor_normalizes_hex_string_to_canonical(self):
        """A bare 32-char hex string (no dashes) must be coerced to canonical form."""
        process = DatabricksUUID().bind_processor(self.dialect)
        assert process(self.HEX) == self.HYPHENATED

    def test_bind_processor_normalizes_hyphenated_string(self):
        """A canonical hyphenated string passes through unchanged."""
        process = DatabricksUUID().bind_processor(self.dialect)
        assert process(self.HYPHENATED) == self.HYPHENATED

    def test_bind_processor_rejects_non_uuid_string(self):
        """Bad input must raise instead of being silently written to the column."""
        process = DatabricksUUID().bind_processor(self.dialect)
        with pytest.raises(ValueError):
            process("not-a-uuid")

    def test_literal_processor_rejects_injection_attempt(self):
        """An attacker-controlled string must not be allowed to escape the quotes.

        Before the input was normalized through ``UUID(...)``, a string like
        ``abc' OR '1'='1`` would inject directly into ``WHERE id = 'abc' OR
        '1'='1'`` whenever ``literal_binds=True`` was used.
        """
        process = DatabricksUUID().literal_processor(self.dialect)
        with pytest.raises(ValueError):
            process("abc' OR '1'='1")

    def test_literal_processor_rejects_injection_via_uuid_subclass(self):
        """A UUID subclass with a malicious ``__str__`` must not bypass escaping.

        ``_canonical`` reconstructs every UUID through ``UUID(int=value.int)``
        so a subclass cannot inject SQL via an overridden string conversion.
        """

        class EvilUUID(UUID):
            def __str__(self):  # type: ignore[override]
                return "abc' OR '1'='1"

        process = DatabricksUUID().literal_processor(self.dialect)
        rendered = process(EvilUUID(self.HYPHENATED))
        assert rendered == "'%s'" % self.HYPHENATED
        assert "OR" not in rendered

    def test_literal_processor_rejects_injection_with_as_uuid_false(self):
        """``Uuid(as_uuid=False)`` shares ``_canonical``; lock in coverage anyway."""
        process = DatabricksUUID(as_uuid=False).literal_processor(self.dialect)
        with pytest.raises(ValueError):
            process("abc' OR '1'='1")

    def test_as_uuid_false_round_trip_normalizes_hex_input(self):
        """``Uuid(as_uuid=False)`` users sometimes pass hex-form strings.

        Pre-fix, the dialect wrote those through unchanged, so a query against
        canonically-stored data returned 0 rows. After the fix, both forms
        normalize to canonical on the wire.
        """
        type_ = DatabricksUUID(as_uuid=False)
        bind = type_.bind_processor(self.dialect)
        result = type_.result_processor(self.dialect, None)

        assert bind(self.HEX) == self.HYPHENATED
        assert bind(self.HYPHENATED) == self.HYPHENATED
        assert result(self.HYPHENATED) == self.HYPHENATED
        assert result(self.HEX) == self.HYPHENATED
