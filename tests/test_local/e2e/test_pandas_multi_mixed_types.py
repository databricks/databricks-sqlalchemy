import uuid
import json
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pandas as pd
import pytest
from sqlalchemy import Integer, Uuid, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError

from databricks.sqlalchemy import DatabricksVariant


@pytest.fixture
def db_engine(connection_details) -> Engine:
    host = connection_details["host"]
    http_path = connection_details["http_path"]
    access_token = connection_details["access_token"]
    catalog = connection_details["catalog"]
    schema = connection_details["schema"]

    conn_string = (
        f"databricks://token:{access_token}@{host}"
        f"?http_path={http_path}&catalog={catalog}&schema={schema}"
    )
    engine = create_engine(
        conn_string, connect_args={"_user_agent_entry": "SQLAlchemy pandas e2e tests"}
    )
    try:
        yield engine
    finally:
        engine.dispose()


def test_pandas_to_sql_multi_mixed_object_column_succeeds(db_engine: Engine):
    table_name = f"pecoblr_2746_e2e_{uuid.uuid4().hex[:8]}"
    fq_table_name = f"`main`.`default`.`{table_name}`"
    df = pd.DataFrame(
        {
            "name": ["alice", "bob", None],
            "value": [1, 0, "NE"],
            "score": [9.5, 8.1, None],
            "active": [True, None, False],
        }
    )

    try:
        with db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {fq_table_name}"))
            conn.execute(
                text(
                    f"CREATE TABLE {fq_table_name} "
                    "(name STRING, value STRING, score DOUBLE, active BOOLEAN) "
                    "USING DELTA"
                )
            )

        # This is the failing path from PECOBLR-2746 before the adaptive cast fix.
        df.to_sql(
            table_name,
            db_engine,
            schema="default",
            if_exists="append",
            index=False,
            method="multi",
        )

        with db_engine.begin() as conn:
            rows = conn.execute(
                text(
                    f"SELECT name, value, score, active FROM {fq_table_name} "
                    "ORDER BY CASE WHEN name IS NULL THEN 1 ELSE 0 END, name"
                )
            ).fetchall()

        assert len(rows) == 3
        assert rows[0][0] == "alice"
        assert rows[0][1] == "1"
        assert rows[0][2] == pytest.approx(9.5)
        assert rows[0][3] is True

        assert rows[1][0] == "bob"
        assert rows[1][1] == "0"
        assert rows[1][2] == pytest.approx(8.1)
        assert rows[1][3] is None

        assert rows[2][0] is None
        assert rows[2][1] == "NE"
        assert rows[2][2] is None
        assert rows[2][3] is False
    finally:
        with db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {fq_table_name}"))


def test_pandas_to_sql_multi_example_types_succeeds(db_engine: Engine):
    table_name = f"pecoblr_2746_example_types_{uuid.uuid4().hex[:8]}"
    fq_table_name = f"`main`.`default`.`{table_name}`"
    base_variant = {
        "name": "John Doe",
        "age": 30,
        "address": {"city": "San Francisco", "state": "CA"},
        "hobbies": ["reading", "hiking"],
        "is_active": True,
    }
    rows = [
        {
            "bigint_col": 1234567890123456789,
            "string_col": "foo",
            "tinyint_col": -100,
            "int_col": 5280,
            "numeric_col": Decimal("525600.01"),
            "boolean_col": True,
            "date_col": date(2020, 12, 25),
            "datetime_col": datetime(
                1991, 8, 3, 21, 30, 5, tzinfo=timezone(timedelta(hours=-8))
            ),
            "datetime_col_ntz": datetime(1990, 12, 4, 6, 33, 41),
            "time_col": time(23, 59, 59),
            "uuid_col": UUID(int=255),
            "variant_col": base_variant,
        },
        {
            "bigint_col": 2234567890123456789,
            "string_col": "bar",
            "tinyint_col": 100,
            "int_col": 42,
            "numeric_col": Decimal("123.45"),
            "boolean_col": False,
            "date_col": date(2021, 1, 2),
            "datetime_col": datetime(
                1992, 9, 4, 22, 31, 6, tzinfo=timezone(timedelta(hours=-7))
            ),
            "datetime_col_ntz": datetime(1991, 1, 5, 7, 34, 42),
            "time_col": time(1, 2, 3),
            "uuid_col": UUID(int=256),
            "variant_col": base_variant | {"name": "Jane Doe"},
        },
    ]
    df = pd.DataFrame(rows)

    try:
        with db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {fq_table_name}"))
            conn.execute(
                text(
                    f"CREATE TABLE {fq_table_name} ("
                    "bigint_col BIGINT, "
                    "string_col STRING, "
                    "tinyint_col TINYINT, "
                    "int_col INT, "
                    "numeric_col DECIMAL(10, 2), "
                    "boolean_col BOOLEAN, "
                    "date_col DATE, "
                    "datetime_col TIMESTAMP, "
                    "datetime_col_ntz TIMESTAMP_NTZ, "
                    "time_col STRING, "
                    "uuid_col STRING, "
                    "variant_col VARIANT"
                    ") USING DELTA"
                )
            )

        df.to_sql(
            table_name,
            db_engine,
            schema="default",
            if_exists="append",
            index=False,
            method="multi",
            dtype={"uuid_col": Uuid(), "variant_col": DatabricksVariant()},
        )

        with db_engine.begin() as conn:
            result = conn.execute(
                text(
                    f"SELECT bigint_col, string_col, tinyint_col, int_col, "
                    f"numeric_col, boolean_col, date_col, datetime_col, "
                    f"datetime_col_ntz, time_col, uuid_col, TO_JSON(variant_col) "
                    f"FROM {fq_table_name} ORDER BY bigint_col"
                )
            ).fetchall()

        assert len(result) == 2
        assert result[0][0] == rows[0]["bigint_col"]
        assert result[0][1] == rows[0]["string_col"]
        assert result[0][2] == rows[0]["tinyint_col"]
        assert result[0][3] == rows[0]["int_col"]
        assert result[0][4] == rows[0]["numeric_col"]
        assert result[0][5] is rows[0]["boolean_col"]
        assert result[0][6] == rows[0]["date_col"]
        assert result[0][8] == rows[0]["datetime_col_ntz"]
        assert result[0][9] == "23:59:59"
        assert result[0][10] == str(rows[0]["uuid_col"])
        assert json.loads(result[0][11]) == rows[0]["variant_col"]

        assert result[1][0] == rows[1]["bigint_col"]
        assert result[1][1] == rows[1]["string_col"]
        assert result[1][2] == rows[1]["tinyint_col"]
        assert result[1][3] == rows[1]["int_col"]
        assert result[1][4] == rows[1]["numeric_col"]
        assert result[1][5] is rows[1]["boolean_col"]
        assert result[1][6] == rows[1]["date_col"]
        assert result[1][8] == rows[1]["datetime_col_ntz"]
        assert result[1][9] == "01:02:03"
        assert result[1][10] == str(rows[1]["uuid_col"])
        assert json.loads(result[1][11]) == rows[1]["variant_col"]
    finally:
        with db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {fq_table_name}"))


def test_pandas_to_sql_multi_mixed_scalar_families_cast_to_string(db_engine: Engine):
    table_name = f"pecoblr_2746_scalar_families_{uuid.uuid4().hex[:8]}"
    fq_table_name = f"`main`.`default`.`{table_name}`"
    df = pd.DataFrame(
        {
            "number_value": [1, "one"],
            "decimal_value": [Decimal("1.25"), "one point two five"],
            "boolean_value": [True, "true"],
            "date_value": [date(2020, 12, 25), "christmas"],
            "datetime_value": [datetime(1990, 12, 4, 6, 33, 41), "datetime"],
            "uuid_value": [UUID(int=255), str(UUID(int=256))],
        }
    )

    try:
        with db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {fq_table_name}"))
            conn.execute(
                text(
                    f"CREATE TABLE {fq_table_name} ("
                    "number_value STRING, "
                    "decimal_value STRING, "
                    "boolean_value STRING, "
                    "date_value STRING, "
                    "datetime_value STRING, "
                    "uuid_value STRING"
                    ") USING DELTA"
                )
            )

        df.to_sql(
            table_name,
            db_engine,
            schema="default",
            if_exists="append",
            index=False,
            method="multi",
            dtype={"uuid_value": Uuid()},
        )

        with db_engine.begin() as conn:
            rows = conn.execute(
                text(
                    f"SELECT number_value, decimal_value, boolean_value, date_value, "
                    f"datetime_value, uuid_value FROM {fq_table_name} "
                    "ORDER BY number_value"
                )
            ).fetchall()

        assert len(rows) == 2
        assert rows[0][0] == "1"
        assert rows[0][1] == "1.25"
        assert rows[0][2].lower() == "true"
        assert rows[0][3] == "2020-12-25"
        assert rows[0][5] == str(UUID(int=255))
        assert rows[1] == (
            "one",
            "one point two five",
            "true",
            "christmas",
            "datetime",
            str(UUID(int=256)),
        )
    finally:
        with db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {fq_table_name}"))


def test_pandas_to_sql_multi_mixed_scalar_non_string_target_fails_loudly(
    db_engine: Engine,
):
    table_name = f"pecoblr_2746_non_string_target_{uuid.uuid4().hex[:8]}"
    fq_table_name = f"`main`.`default`.`{table_name}`"
    df = pd.DataFrame({"value": [1, "not-a-number"]})

    try:
        with db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {fq_table_name}"))
            conn.execute(text(f"CREATE TABLE {fq_table_name} (value INT) USING DELTA"))

        with pytest.raises(DatabaseError) as exc_info:
            df.to_sql(
                table_name,
                db_engine,
                schema="default",
                if_exists="append",
                index=False,
                method="multi",
                dtype={"value": Integer()},
            )

        assert "INVALID_INLINE_TABLE.INCOMPATIBLE_TYPES_IN_INLINE_TABLE" in str(
            exc_info.value
        )
    finally:
        with db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {fq_table_name}"))
