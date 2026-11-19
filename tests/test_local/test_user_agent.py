import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine.default import DefaultDialect

from databricks.sqlalchemy import DatabricksDialect
from databricks.sqlalchemy.base import receive_do_connect


@pytest.mark.parametrize(
    "connect_args, expected",
    [
        ({}, "sqlalchemy"),
        ({"user_agent_entry": ""}, "sqlalchemy"),
        ({"user_agent_entry": None}, "sqlalchemy"),
        ({"user_agent_entry": "example-client"}, "sqlalchemy + example-client"),
        ({"user_agent_entry": "sqlalchemy"}, "sqlalchemy"),
        (
            {"user_agent_entry": "sqlalchemy + example-client"},
            "sqlalchemy + example-client",
        ),
        (
            {"user_agent_entry": "example-sqlalchemy-client"},
            "example-sqlalchemy-client",
        ),
        ({"_user_agent_entry": ""}, "sqlalchemy"),
        ({"_user_agent_entry": None}, "sqlalchemy"),
        ({"_user_agent_entry": "example-client"}, "sqlalchemy + example-client"),
        ({"_user_agent_entry": "sqlalchemy"}, "sqlalchemy"),
        (
            {"_user_agent_entry": "legacy-client", "user_agent_entry": "modern-client"},
            "sqlalchemy + legacy-client",
        ),
        (
            {"_user_agent_entry": "", "user_agent_entry": "modern-client"},
            "sqlalchemy + modern-client",
        ),
        (
            {"_user_agent_entry": None, "user_agent_entry": None},
            "sqlalchemy",
        ),
    ],
)
def test_user_agent_tag_is_idempotent(connect_args, expected):
    dialect = DatabricksDialect()
    cparams = {"http_path": "/sql/1.0/warehouses/example", **connect_args}

    for _ in range(5):
        receive_do_connect(dialect, None, [], cparams)
        assert cparams == {
            "http_path": "/sql/1.0/warehouses/example",
            "user_agent_entry": expected,
        }


def test_other_dialects_keep_connect_parameters():
    cparams = {"_user_agent_entry": "example-client", "user_agent_entry": ""}
    original = cparams.copy()

    receive_do_connect(DefaultDialect(), None, [], cparams)

    assert cparams == original


@pytest.mark.parametrize("connect_args", [{}, {"user_agent_entry": None}])
def test_engine_connection_attempts_use_one_sqlalchemy_tag(monkeypatch, connect_args):
    engine = create_engine(
        "databricks://token:unused@example.invalid?http_path=/sql/1.0/warehouses/example",
        connect_args=connect_args,
    )
    observed = []

    class StopBeforeNetwork(Exception):
        pass

    def capture_connect(*args, **kwargs):
        observed.append(kwargs["user_agent_entry"])
        raise StopBeforeNetwork

    # Stop at the DBAPI boundary so the real engine event runs without a warehouse.
    monkeypatch.setattr(engine.dialect.loaded_dbapi, "connect", capture_connect)
    try:
        for _ in range(3):
            with pytest.raises(StopBeforeNetwork):
                engine.connect()
    finally:
        engine.dispose()

    assert observed == ["sqlalchemy"] * 3
