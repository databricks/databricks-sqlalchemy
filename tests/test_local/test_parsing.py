import pytest
from databricks.sqlalchemy._parse import (
    extract_identifiers_from_string,
    extract_identifier_groups_from_string,
    extract_three_level_identifier_from_constraint_string,
    build_fk_dict,
    build_pk_dict,
    match_dte_rows_by_value,
    get_comment_from_dte_output,
    DatabricksSqlAlchemyParseException,
)
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Integer,
    Numeric,
    String,
    Time,
    Uuid,
)

from databricks.sqlalchemy import (
    DatabricksArray,
    TIMESTAMP,
    TINYINT,
    DatabricksMap,
    TIMESTAMP_NTZ,
)
from databricks.sqlalchemy import DatabricksDialect

dialect = DatabricksDialect()

# These are outputs from DESCRIBE TABLE EXTENDED
@pytest.mark.parametrize(
    "input, expected",
    [
        ("PRIMARY KEY (`pk1`, `pk2`)", ["pk1", "pk2"]),
        ("PRIMARY KEY (`a`, `b`, `c`)", ["a", "b", "c"]),
        ("PRIMARY KEY (`name`, `id`, `attr`)", ["name", "id", "attr"]),
    ],
)
def test_extract_identifiers(input, expected):
    assert (
        extract_identifiers_from_string(input) == expected
    ), "Failed to extract identifiers from string"


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            "FOREIGN KEY (`pname`, `pid`, `pattr`) REFERENCES `main`.`pysql_sqlalchemy`.`tb1` (`name`, `id`, `attr`)",
            [
                "(`pname`, `pid`, `pattr`)",
                "(`name`, `id`, `attr`)",
            ],
        )
    ],
)
def test_extract_identifer_batches(input, expected):
    assert (
        extract_identifier_groups_from_string(input) == expected
    ), "Failed to extract identifier groups from string"


def test_extract_3l_namespace_from_constraint_string():
    input = "FOREIGN KEY (`parent_user_id`) REFERENCES `main`.`pysql_dialect_compliance`.`users` (`user_id`)"
    expected = {
        "catalog": "main",
        "schema": "pysql_dialect_compliance",
        "table": "users",
    }

    assert (
        extract_three_level_identifier_from_constraint_string(input) == expected
    ), "Failed to extract 3L namespace from constraint string"


def test_extract_3l_namespace_from_bad_constraint_string():
    input = "FOREIGN KEY (`parent_user_id`) REFERENCES `pysql_dialect_compliance`.`users` (`user_id`)"

    with pytest.raises(DatabricksSqlAlchemyParseException):
        extract_three_level_identifier_from_constraint_string(input)


@pytest.mark.parametrize("tschema", [None, "some_schema"])
def test_build_fk_dict(tschema):
    fk_constraint_string = "FOREIGN KEY (`parent_user_id`) REFERENCES `main`.`some_schema`.`users` (`user_id`)"

    result = build_fk_dict("some_fk_name", fk_constraint_string, schema_name=tschema)

    assert result == {
        "name": "some_fk_name",
        "constrained_columns": ["parent_user_id"],
        "referred_schema": tschema,
        "referred_table": "users",
        "referred_columns": ["user_id"],
    }


def test_build_pk_dict():
    pk_constraint_string = "PRIMARY KEY (`id`, `name`, `email_address`)"
    pk_name = "pk1"

    result = build_pk_dict(pk_name, pk_constraint_string)

    assert result == {
        "constrained_columns": ["id", "name", "email_address"],
        "name": "pk1",
    }


# This is a real example of the output from DESCRIBE TABLE EXTENDED as of 15 October 2023
RAW_SAMPLE_DTE_OUTPUT = [
    ["id", "int"],
    ["name", "string"],
    ["", ""],
    ["# Detailed Table Information", ""],
    ["Catalog", "main"],
    ["Database", "pysql_sqlalchemy"],
    ["Table", "exampleexampleexample"],
    ["Created Time", "Sun Oct 15 21:12:54 UTC 2023"],
    ["Last Access", "UNKNOWN"],
    ["Created By", "Spark "],
    ["Type", "MANAGED"],
    ["Location", "s3://us-west-2-****-/19a85dee-****/tables/ccb7***"],
    ["Provider", "delta"],
    ["Comment", "some comment"],
    ["Owner", "some.user@example.com"],
    ["Is_managed_location", "true"],
    ["Predictive Optimization", "ENABLE (inherited from CATALOG main)"],
    [
        "Table Properties",
        "[delta.checkpoint.writeStatsAsJson=false,delta.checkpoint.writeStatsAsStruct=true,delta.minReaderVersion=1,delta.minWriterVersion=2]",
    ],
    ["", ""],
    ["# Constraints", ""],
    ["exampleexampleexample_pk", "PRIMARY KEY (`id`)"],
    [
        "exampleexampleexample_fk",
        "FOREIGN KEY (`parent_user_id`) REFERENCES `main`.`pysql_dialect_compliance`.`users` (`user_id`)",
    ],
]

FMT_SAMPLE_DT_OUTPUT = [
    {"col_name": i[0], "data_type": i[1]} for i in RAW_SAMPLE_DTE_OUTPUT
]


@pytest.mark.parametrize(
    "match, output",
    [
        (
            "PRIMARY KEY",
            [
                {
                    "col_name": "exampleexampleexample_pk",
                    "data_type": "PRIMARY KEY (`id`)",
                }
            ],
        ),
        (
            "FOREIGN KEY",
            [
                {
                    "col_name": "exampleexampleexample_fk",
                    "data_type": "FOREIGN KEY (`parent_user_id`) REFERENCES `main`.`pysql_dialect_compliance`.`users` (`user_id`)",
                }
            ],
        ),
    ],
)
def test_filter_dict_by_value(match, output):
    result = match_dte_rows_by_value(FMT_SAMPLE_DT_OUTPUT, match)
    assert result == output


def test_get_comment_from_dte_output():
    assert get_comment_from_dte_output(FMT_SAMPLE_DT_OUTPUT) == "some comment"


def get_databricks_non_compound_types():
    return [
        Integer(),
        String(),
        Boolean(),
        Date(),
        DateTime(),
        Time(),
        Uuid(),
        Numeric(),
        TINYINT(),
        TIMESTAMP(),
        TIMESTAMP_NTZ(),
        BigInteger(),
    ]


def get_databricks_compound_types():
    return [DatabricksArray(String), DatabricksMap(String, String)]


@pytest.mark.parametrize("internal_type", get_databricks_non_compound_types())
def test_array_parsing(internal_type):
    array_type = DatabricksArray(internal_type)

    actual_parsed = array_type.compile(dialect=dialect)
    expected_parsed = "ARRAY<{}>".format(internal_type.compile(dialect=dialect))
    assert actual_parsed == expected_parsed


@pytest.mark.parametrize("internal_type_1", get_databricks_non_compound_types())
@pytest.mark.parametrize("internal_type_2", get_databricks_non_compound_types())
def test_map_parsing(internal_type_1, internal_type_2):
    map_type = DatabricksMap(internal_type_1, internal_type_2)

    actual_parsed = map_type.compile(dialect=dialect)
    expected_parsed = "MAP<{},{}>".format(
        internal_type_1.compile(dialect=dialect),
        internal_type_2.compile(dialect=dialect),
    )
    assert actual_parsed == expected_parsed


@pytest.mark.parametrize(
    "internal_type",
    get_databricks_non_compound_types() + get_databricks_compound_types(),
)
def test_multilevel_array_type_parsing(internal_type):
    array_type = DatabricksArray(DatabricksArray(DatabricksArray(internal_type)))

    actual_parsed = array_type.compile(dialect=dialect)
    expected_parsed = "ARRAY<ARRAY<ARRAY<{}>>>".format(
        internal_type.compile(dialect=dialect)
    )
    assert actual_parsed == expected_parsed


@pytest.mark.parametrize(
    "internal_type",
    get_databricks_non_compound_types() + get_databricks_compound_types(),
)
def test_multilevel_map_type_parsing(internal_type):
    map_type = DatabricksMap(
        String, DatabricksMap(String, DatabricksMap(String, internal_type))
    )

    actual_parsed = map_type.compile(dialect=dialect)
    expected_parsed = "MAP<STRING,MAP<STRING,MAP<STRING,{}>>>".format(
        internal_type.compile(dialect=dialect)
    )
    assert actual_parsed == expected_parsed
