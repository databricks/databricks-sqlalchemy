import os
import time
import uuid

import pytest
from sqlalchemy import (
    BigInteger,
    Identity,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    inspect,
    insert,
    select,
    text,
)
from sqlalchemy.dialects import registry
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from databricks.sqlalchemy.base import DatabricksDialect

registry.register("databricks", "databricks.sqlalchemy", "DatabricksDialect")

REQUIRED_ENV = [
    "DATABRICKS_SERVER_HOSTNAME",
    "DATABRICKS_HTTP_PATH",
    "DATABRICKS_CATALOG",
    "DATABRICKS_SCHEMA",
    "DATABRICKS_SP_CLIENT_ID",
    "DATABRICKS_SP_CLIENT_SECRET",
]


def _load_env():
    env = {key: os.getenv(key) for key in REQUIRED_ENV}
    missing = [key for key, value in env.items() if not value]
    if missing:
        pytest.skip(
            f"Service principal env vars missing: {', '.join(missing)}. "
            "Populate test.env with workspace and service principal credentials."
        )
    return env


def _build_url(env, extra_params: str = "") -> str:
    base = (
        "databricks://"
        f"{env['DATABRICKS_SP_CLIENT_ID']}:{env['DATABRICKS_SP_CLIENT_SECRET']}"
        f"@{env['DATABRICKS_SERVER_HOSTNAME']}"
        f"?http_path={env['DATABRICKS_HTTP_PATH']}"
        f"&catalog={env['DATABRICKS_CATALOG']}"
        f"&schema={env['DATABRICKS_SCHEMA']}"
        "&authentication=service_principal"
    )
    if extra_params:
        base += "&" + extra_params.lstrip("&")
    return base


def _fully_qualified(env, table_name: str) -> str:
    return (
        f"`{env['DATABRICKS_CATALOG']}`."
        f"`{env['DATABRICKS_SCHEMA']}`."
        f"`{table_name}`"
    )


def _random_table_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def sp_env():
    return _load_env()


@pytest.fixture
def sp_engine_factory(sp_env):
    engines = []

    def factory(extra_params: str = ""):
        engine = create_engine(_build_url(sp_env, extra_params))
        engines.append(engine)
        return engine

    yield factory

    for engine in engines:
        engine.dispose()


def _drop_table(engine, fq_name: str):
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {fq_name}"))


def _create_identity_table(engine, fq_name: str):
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE {fq_name} (
                    id BIGINT GENERATED ALWAYS AS IDENTITY,
                    value STRING
                )
                """
            )
        )


def test_sp_basic_crud(sp_env, sp_engine_factory):
    engine = sp_engine_factory()
    table_name = _random_table_name("sp_basic")
    fq_name = _fully_qualified(sp_env, table_name)

    try:
        _drop_table(engine, fq_name)
        _create_identity_table(engine, fq_name)
        with engine.begin() as conn:
            conn.execute(
                text(f"INSERT INTO {fq_name} (value) VALUES (:value)"),
                {"value": "hello"},
            )
            rows = conn.execute(text(f"SELECT COUNT(*) FROM {fq_name}")).scalar_one()
            assert rows == 1

            conn.execute(
                text(f"DELETE FROM {fq_name} WHERE value = :value"),
                {"value": "hello"},
            )
            rows = conn.execute(text(f"SELECT COUNT(*) FROM {fq_name}")).scalar_one()
            assert rows == 0
    finally:
        _drop_table(engine, fq_name)


def test_sp_scope_override(sp_env, sp_engine_factory):
    engine = sp_engine_factory("sp_scopes=sql")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar_one()
            assert result == 1
    finally:
        engine.dispose()


def test_sp_reflection(sp_env, sp_engine_factory):
    engine = sp_engine_factory()
    table_name = _random_table_name("sp_reflect")
    fq_name = _fully_qualified(sp_env, table_name)

    try:
        _drop_table(engine, fq_name)
        _create_identity_table(engine, fq_name)
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name, schema=sp_env["DATABRICKS_SCHEMA"])
        id_column = next(col for col in columns if col["name"] == "id")
        assert id_column["type"].__class__.__name__.lower().startswith("bigint")
    finally:
        _drop_table(engine, fq_name)


def test_sp_token_refresh_allows_followup_insert(
    sp_env, sp_engine_factory, monkeypatch
):
    captured = {}
    original = DatabricksDialect._build_service_principal_provider

    def wrapper(self, url):
        provider = original(self, url)
        captured["provider"] = provider
        return provider

    monkeypatch.setattr(DatabricksDialect, "_build_service_principal_provider", wrapper)
    engine = sp_engine_factory()
    table_name = _random_table_name("sp_refresh")
    fq_name = _fully_qualified(sp_env, table_name)

    try:
        _drop_table(engine, fq_name)
        _create_identity_table(engine, fq_name)
        with engine.begin() as conn:
            conn.execute(
                text(f"INSERT INTO {fq_name} (value) VALUES (:value)"),
                {"value": "first"},
            )

        provider = captured.get("provider")
        assert provider is not None, "credentials provider was not captured"
        provider._expires_at = time.time() - 10

        with engine.begin() as conn:
            conn.execute(
                text(f"INSERT INTO {fq_name} (value) VALUES (:value)"),
                {"value": "second"},
            )
            rows = conn.execute(text(f"SELECT COUNT(*) FROM {fq_name}")).scalar_one()
            assert rows == 2
    finally:
        _drop_table(engine, fq_name)


def test_sp_orm_identity_roundtrip(sp_env, sp_engine_factory):
    engine = sp_engine_factory()
    table_name = _random_table_name("sp_orm")

    class Base(DeclarativeBase):
        pass

    schema = sp_env["DATABRICKS_SCHEMA"]

    class OrmRecord(Base):
        __tablename__ = table_name
        __table_args__ = {"schema": schema}

        id: Mapped[int] = mapped_column(
            BigInteger, Identity(always=True), primary_key=True
        )
        value: Mapped[str] = mapped_column(String(100))

    try:
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            result = session.execute(insert(OrmRecord).values(value="orm-test"))
            session.commit()

        with Session(engine) as session:
            inserted_id = session.scalar(
                select(OrmRecord.id)
                .where(OrmRecord.value == "orm-test")
                .order_by(OrmRecord.id.desc())
                .limit(1)
            )
            assert inserted_id is not None

            fetched = session.get(OrmRecord, inserted_id)
            assert fetched is not None
            assert fetched.value == "orm-test"

            session.delete(fetched)
            session.commit()

            remaining = session.scalar(select(func.count()).select_from(OrmRecord))
            assert remaining == 0
    finally:
        Base.metadata.drop_all(engine)
