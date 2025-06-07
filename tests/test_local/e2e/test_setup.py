import pytest
from sqlalchemy import create_engine, Engine
from contextlib import contextmanager
from sqlalchemy.orm import DeclarativeBase, Session


class TestSetup:
    @pytest.fixture(autouse=True)
    def get_details(self, connection_details):
        self.arguments = connection_details.copy()

    def db_engine(self) -> Engine:
        HOST = self.arguments["host"]
        HTTP_PATH = self.arguments["http_path"]
        ACCESS_TOKEN = self.arguments["access_token"]
        CATALOG = self.arguments["catalog"]
        SCHEMA = self.arguments["schema"]

        connect_args = {"_user_agent_entry": "SQLAlchemy e2e Tests"}

        conn_string = f"databricks://token:{ACCESS_TOKEN}@{HOST}?http_path={HTTP_PATH}&catalog={CATALOG}&schema={SCHEMA}"
        return create_engine(conn_string, connect_args=connect_args)

    @contextmanager
    def table_context(self, table: DeclarativeBase):
        engine = self.db_engine()
        table.metadata.create_all(engine)
        try:
            yield engine
        finally:
            table.metadata.drop_all(engine)
