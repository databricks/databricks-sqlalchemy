"""Typed ARRAY/MAP and binary values bind as JSON / hex strings rebuilt in SQL."""

from datetime import date, datetime
from decimal import Decimal

import sqlalchemy as sa

from databricks.sqlalchemy import DatabricksArray, DatabricksMap

engine = sa.create_engine("databricks://token:x@host?http_path=p")
dialect = engine.dialect


def compile_insert(*columns):
    table = sa.Table("t", sa.MetaData(), *columns)
    return str(table.insert().compile(dialect=dialect))


def processed(type_, value):
    return type_.dialect_impl(dialect).bind_processor(dialect)(value)


def test_array_renders_from_json():
    sql = compile_insert(sa.Column("a", DatabricksArray(sa.Numeric(38, 18))))
    assert "from_json(:`a`, 'ARRAY<DECIMAL(38, 18)>', map('mode', 'FAILFAST'))" in sql
    assert "CAST" not in sql


def test_map_is_parsed_with_string_keys_and_cast():
    sql = compile_insert(sa.Column("m", DatabricksMap(sa.Integer, sa.String)))
    assert (
        "CAST(from_json(:`m`, 'MAP<STRING, STRING>', map('mode', 'FAILFAST')) "
        "AS MAP<INT,STRING>)"
    ) in sql


def test_values_serialize_exactly():
    assert (
        processed(DatabricksArray(sa.Numeric(38, 18)), [Decimal("1E-18"), None])
        == "[0.000000000000000001,null]"
    )
    assert processed(DatabricksArray(sa.BigInteger), [9007199254740993]) == (
        "[9007199254740993]"
    )
    assert processed(DatabricksMap(sa.Integer, sa.String), {7: "雪"}) == '{"7":"雪"}'
    assert processed(DatabricksArray(sa.Date), [date(2024, 2, 29)]) == (
        '["2024-02-29"]'
    )
    assert processed(DatabricksArray(sa.DateTime), [datetime(2024, 1, 1, 1, 2, 3, 4)]) == (
        '["2024-01-01T01:02:03.000004"]'
    )
    assert processed(DatabricksArray(sa.String), None) is None


def test_binary_binds_as_hex_with_unhex():
    sql = compile_insert(sa.Column("b", sa.LargeBinary))
    assert "unhex(:`b`)" in sql
    assert processed(sa.LargeBinary(), b"\x00\xff") == "00ff"


def test_element_bind_processors_are_applied():
    import uuid
    from datetime import time

    u = uuid.UUID(int=1)
    assert processed(DatabricksArray(sa.Uuid), [u, None]) == f'["{u}",null]'
    assert processed(DatabricksArray(sa.Time), [time(1, 2, 3)]) == '["01:02:03"]'
    assert processed(DatabricksMap(sa.Uuid, sa.Integer), {u: 1}) == f'{{"{u}":1}}'
    # exact numeric serialization is unchanged (Numeric has no bind processor here)
    assert processed(DatabricksArray(sa.Numeric(38, 18)), [Decimal("1E-18")]) == (
        "[0.000000000000000001]"
    )
