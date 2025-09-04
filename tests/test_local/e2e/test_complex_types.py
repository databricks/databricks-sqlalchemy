from .test_setup import TestSetup
from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Integer,
    Numeric,
    Boolean,
    Date,
    TIMESTAMP,
    DateTime,
)
from collections.abc import Sequence
from databricks.sqlalchemy import TIMESTAMP, TINYINT, DatabricksArray, DatabricksMap, DatabricksVariant
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy import select
from datetime import date, datetime, time, timedelta, timezone
import pandas as pd
import numpy as np
import decimal
import json

class TestComplexTypes(TestSetup):
    def _parse_to_common_type(self, value):
        """
        Function to convert the :value passed into a common python datatype for comparison

        Convertion fyi
        MAP Datatype on server is returned as a list of tuples
            Ex:
                {"a":1,"b":2} -> [("a",1),("b",2)]

        ARRAY Datatype on server is returned as a numpy array
            Ex:
                ["a","b","c"] -> np.array(["a","b","c"],dtype=object)

        Primitive datatype on server is returned as a numpy primitive
            Ex:
                1 -> np.int64(1)
                2 -> np.int32(2)
        """
        if value is None:
            return None
        elif isinstance(value, (Sequence, np.ndarray)) and not isinstance(
            value, (str, bytes)
        ):
            return tuple(value)
        elif isinstance(value, dict):
            return tuple(sorted(value.items()))
        elif isinstance(value, np.generic):
            return value.item()
        elif isinstance(value, decimal.Decimal):
            return float(value)
        else:
            return value

    def _recursive_compare(self, actual, expected):
        """
        Function to compare the :actual and :expected values, recursively checks and ensures that all the data matches till the leaf level

        Note: Complex datatype like MAP is not returned as a dictionary but as a list of tuples
        """
        actual_parsed = self._parse_to_common_type(actual)
        expected_parsed = self._parse_to_common_type(expected)

        # Check if types are the same
        if type(actual_parsed) != type(expected_parsed):
            return False

        # Handle lists or tuples
        if isinstance(actual_parsed, (list, tuple)):
            if len(actual_parsed) != len(expected_parsed):
                return False
            return all(
                self._recursive_compare(o1, o2)
                for o1, o2 in zip(actual_parsed, expected_parsed)
            )

        return actual_parsed == expected_parsed

    def sample_array_table(self) -> tuple[DeclarativeBase, dict]:
        class Base(DeclarativeBase):
            pass

        class ArrayTable(Base):
            __tablename__ = "sqlalchemy_array_table"

            int_col = Column(Integer, primary_key=True)
            array_int_col = Column(DatabricksArray(Integer))
            array_bigint_col = Column(DatabricksArray(BigInteger))
            array_numeric_col = Column(DatabricksArray(Numeric(10, 2)))
            array_string_col = Column(DatabricksArray(String))
            array_boolean_col = Column(DatabricksArray(Boolean))
            array_date_col = Column(DatabricksArray(Date))
            array_datetime_col = Column(DatabricksArray(TIMESTAMP))
            array_datetime_col_ntz = Column(DatabricksArray(DateTime))
            array_tinyint_col = Column(DatabricksArray(TINYINT))

        sample_data = {
            "int_col": 1,
            "array_int_col": [1, 2],
            "array_bigint_col": [1234567890123456789, 2345678901234567890],
            "array_numeric_col": [1.1, 2.2],
            "array_string_col": ["a", "b"],
            "array_boolean_col": [True, False],
            "array_date_col": [date(2020, 12, 25), date(2021, 1, 2)],
            "array_datetime_col": [
                datetime(1991, 8, 3, 21, 30, 5, tzinfo=timezone(timedelta(hours=-8))),
                datetime(1991, 8, 3, 21, 30, 5, tzinfo=timezone(timedelta(hours=-8))),
            ],
            "array_datetime_col_ntz": [
                datetime(1990, 12, 4, 6, 33, 41),
                datetime(1990, 12, 4, 6, 33, 41),
            ],
            "array_tinyint_col": [-100, 100],
        }

        return ArrayTable, sample_data

    def sample_map_table(self) -> tuple[DeclarativeBase, dict]:
        class Base(DeclarativeBase):
            pass

        class MapTable(Base):
            __tablename__ = "sqlalchemy_map_table"

            int_col = Column(Integer, primary_key=True)
            map_int_col = Column(DatabricksMap(Integer, Integer))
            map_bigint_col = Column(DatabricksMap(Integer, BigInteger))
            map_numeric_col = Column(DatabricksMap(Integer, Numeric(10, 2)))
            map_string_col = Column(DatabricksMap(Integer, String))
            map_boolean_col = Column(DatabricksMap(Integer, Boolean))
            map_date_col = Column(DatabricksMap(Integer, Date))
            map_datetime_col = Column(DatabricksMap(Integer, TIMESTAMP))
            map_datetime_col_ntz = Column(DatabricksMap(Integer, DateTime))
            map_tinyint_col = Column(DatabricksMap(Integer, TINYINT))

        sample_data = {
            "int_col": 1,
            "map_int_col": {1: 1},
            "map_bigint_col": {1: 1234567890123456789},
            "map_numeric_col": {1: 1.1},
            "map_string_col": {1: "a"},
            "map_boolean_col": {1: True},
            "map_date_col": {1: date(2020, 12, 25)},
            "map_datetime_col": {
                1: datetime(1991, 8, 3, 21, 30, 5, tzinfo=timezone(timedelta(hours=-8)))
            },
            "map_datetime_col_ntz": {1: datetime(1990, 12, 4, 6, 33, 41)},
            "map_tinyint_col": {1: -100},
        }

        return MapTable, sample_data

    def sample_variant_table(self) -> tuple[DeclarativeBase, dict]:
        class Base(DeclarativeBase):
            pass

        class VariantTable(Base):
            __tablename__ = "sqlalchemy_variant_table"

            int_col = Column(Integer, primary_key=True)
            variant_simple_col = Column(DatabricksVariant())
            variant_nested_col = Column(DatabricksVariant())
            variant_array_col = Column(DatabricksVariant())
            variant_mixed_col = Column(DatabricksVariant())

        sample_data = {
            "int_col": 1,
            "variant_simple_col": {"key": "value", "number": 42},
            "variant_nested_col": {"user": {"name": "John", "age": 30}, "active": True},
            "variant_array_col": [1, 2, 3, "hello", {"nested": "data"}],
            "variant_mixed_col": {
                "string": "test",
                "number": 123,
                "boolean": True,
                "array": [1, 2, 3],
                "object": {"nested": "value"}
            }
        }

        return VariantTable, sample_data

    def test_insert_array_table_sqlalchemy(self):
        table, sample_data = self.sample_array_table()

        with self.table_context(table) as engine:
            sa_obj = table(**sample_data)
            session = Session(engine)
            session.add(sa_obj)
            session.commit()

            stmt = select(table).where(table.int_col == 1)

            result = session.scalar(stmt)

            compare = {key: getattr(result, key) for key in sample_data.keys()}
            assert self._recursive_compare(compare, sample_data)

    def test_insert_map_table_sqlalchemy(self):
        table, sample_data = self.sample_map_table()

        with self.table_context(table) as engine:
            sa_obj = table(**sample_data)
            session = Session(engine)
            session.add(sa_obj)
            session.commit()

            stmt = select(table).where(table.int_col == 1)

            result = session.scalar(stmt)

            compare = {key: getattr(result, key) for key in sample_data.keys()}
            assert self._recursive_compare(compare, sample_data)

    def test_array_table_creation_pandas(self):
        table, sample_data = self.sample_array_table()

        with self.table_context(table) as engine:
            # Insert the data into the table
            df = pd.DataFrame([sample_data])
            df.to_sql(table.__tablename__, engine, if_exists="append", index=False)

            # Read the data from the table
            stmt = select(table)
            df_result = pd.read_sql(stmt, engine)
            assert self._recursive_compare(df_result.iloc[0].to_dict(), sample_data)

    def test_map_table_creation_pandas(self):
        table, sample_data = self.sample_map_table()

        with self.table_context(table) as engine:
            # Insert the data into the table
            df = pd.DataFrame([sample_data])
            df.to_sql(table.__tablename__, engine, if_exists="append", index=False)

            # Read the data from the table
            stmt = select(table)
            df_result = pd.read_sql(stmt, engine)
            assert self._recursive_compare(df_result.iloc[0].to_dict(), sample_data)

    def test_insert_variant_table_sqlalchemy(self):
        table, sample_data = self.sample_variant_table()

        with self.table_context(table) as engine:

            sa_obj = table(**sample_data)
            session = Session(engine)
            session.add(sa_obj)
            session.commit()

            stmt = select(table).where(table.int_col == 1)
            result = session.scalar(stmt)
            compare = {key: getattr(result, key) for key in sample_data.keys()}
            # Parse JSON values back to original format for comparison
            for key in ['variant_simple_col', 'variant_nested_col', 'variant_array_col', 'variant_mixed_col']:
                if compare[key] is not None:
                    compare[key] = json.loads(compare[key])

            assert self._recursive_compare(compare, sample_data)

    def test_variant_table_creation_pandas(self):
        table, sample_data = self.sample_variant_table()

        with self.table_context(table) as engine:
            
            df = pd.DataFrame([sample_data])
            dtype_mapping = {
                "variant_simple_col": DatabricksVariant,
                "variant_nested_col": DatabricksVariant,
                "variant_array_col": DatabricksVariant,
                "variant_mixed_col": DatabricksVariant
            }
            df.to_sql(table.__tablename__, engine, if_exists="append", index=False, dtype=dtype_mapping)
            
            stmt = select(table)
            df_result = pd.read_sql(stmt, engine)
            result_dict = df_result.iloc[0].to_dict()
            # Parse JSON values back to original format for comparison
            for key in ['variant_simple_col', 'variant_nested_col', 'variant_array_col', 'variant_mixed_col']:
                if result_dict[key] is not None:
                    result_dict[key] = json.loads(result_dict[key])
            assert self._recursive_compare(result_dict, sample_data)