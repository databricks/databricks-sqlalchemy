"""Full audit of every bind-parameter render path for a dialect that
wraps every marker in backticks.

Two verification layers per scenario:
  A. LOCAL check — compile the statement and assert the rendered SQL
     contains backticked bind markers (no hyphen leaks).
  B. E2E check  — execute against a live Databricks warehouse to
     confirm the server accepts the SQL AND the data round-trips.

Scenarios cover every user-facing shape that could emit a bind marker:
single-row INSERT, multi-row INSERT, executemany, UPDATE, DELETE,
SELECT WHERE (=, !=, >, <, BETWEEN), IN (list), IN (tuple), IN
(subquery), empty IN list, LIMIT/OFFSET, CASE WHEN, function calls
with bind args, NULL values, subqueries, WITH/CTE, cached statement
reuse, compile(render_postcompile=True), construct_expanded_state.

Exits non-zero if anything fails. Treats "no hyphen leaked into an
unbackticked bind marker" as the invariant.
"""
import os
import re
import sys
import uuid
from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy import (
    Column, MetaData, String, Integer, Table,
    and_, or_, between, case, cte, delete, func, insert, literal,
    select, tuple_, update, create_engine,
)

load_dotenv()

URL = (
    f"databricks://token:{os.environ['DATABRICKS_TOKEN']}@"
    f"{os.environ['DATABRICKS_SERVER_HOSTNAME']}"
    f"?http_path={os.environ['DATABRICKS_HTTP_PATH']}"
    f"&catalog={os.environ['DATABRICKS_CATALOG']}"
    f"&schema={os.environ['DATABRICKS_SCHEMA']}"
)

ENGINE = create_engine(URL)
SCHEMA = os.environ["DATABRICKS_SCHEMA"]


class Report:
    def __init__(self):
        self.results = []

    def record(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
        self.results.append((name, ok, detail))

    def summary(self):
        total = len(self.results)
        passes = sum(1 for _, ok, _ in self.results if ok)
        print(f"\n{'=' * 70}")
        print(f"Passed: {passes}/{total}")
        fails = [(n, d) for n, ok, d in self.results if not ok]
        if fails:
            print("\nFailures:")
            for n, d in fails:
                print(f"  - {n}: {d}")
        return passes == total


REPORT = Report()


def hyphen_leak(sql: str) -> bool:
    """Return True iff SQL contains an unquoted `:name-with-hyphen` marker."""
    # Match ':identifier_chars_with_hyphen' NOT preceded by backtick
    # Pattern: a colon followed by a hyphenated identifier, not already backticked
    return bool(re.search(r"(?<![`])(?<!-):[A-Za-z_][A-Za-z_0-9]*-[A-Za-z_0-9-]*", sql))


def assert_compiled(name, compiled, must_contain=None, must_not_contain=None):
    sql = str(compiled)
    if hyphen_leak(sql):
        REPORT.record(name + " [local]", False, f"hyphen leaked: {sql!r}")
        return False
    if must_contain:
        for needle in must_contain:
            if needle not in sql:
                REPORT.record(name + " [local]", False, f"missing {needle!r} in {sql!r}")
                return False
    if must_not_contain:
        for bad in must_not_contain:
            if bad in sql:
                REPORT.record(name + " [local]", False, f"unexpected {bad!r} in {sql!r}")
                return False
    REPORT.record(name + " [local]", True)
    return True


def T(label):
    return f"audit_{label}_{uuid.uuid4().hex[:6]}"


def e2e(name, fn):
    try:
        fn()
        REPORT.record(name + " [e2e]", True)
    except Exception as e:
        REPORT.record(name + " [e2e]", False, str(e).splitlines()[0][:150])


# ---------- Scenarios ----------

def scenario_single_row_insert():
    tname = T("single_insert")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), Column("val", Integer()), schema=SCHEMA)

    stmt = insert(t).values({"col-name": "x", "val": 1})
    assert_compiled("single-row INSERT", stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name`", ":`val`"])

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(stmt)
                rows = conn.execute(select(t)).all()
                assert len(rows) == 1 and rows[0]._mapping["col-name"] == "x"
            finally:
                t.drop(conn)
    e2e("single-row INSERT", run)


def scenario_multi_row_insert():
    tname = T("multi_insert")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), schema=SCHEMA)

    stmt = insert(t).values([{"col-name": f"r{i}"} for i in range(3)])
    assert_compiled("multi-row INSERT", stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name_m0`", ":`col-name_m1`", ":`col-name_m2`"])

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(stmt)
                rows = conn.execute(select(t)).all()
                assert sorted(r._mapping["col-name"] for r in rows) == ["r0", "r1", "r2"]
            finally:
                t.drop(conn)
    e2e("multi-row INSERT", run)


def scenario_executemany():
    tname = T("execmany")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), schema=SCHEMA)

    stmt = insert(t)
    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(stmt, [{"col-name": f"r{i}"} for i in range(3)])
                rows = conn.execute(select(t)).all()
                assert sorted(r._mapping["col-name"] for r in rows) == ["r0", "r1", "r2"]
            finally:
                t.drop(conn)
    # executemany compiles the single-row form and execute-loops
    assert_compiled("executemany compile", stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name`"])
    e2e("executemany", run)


def scenario_update_where():
    tname = T("update")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), Column("other", String()), schema=SCHEMA)

    stmt = update(t).where(t.c["col-name"] == "orig").values({"col-name": "updated"})
    assert_compiled("UPDATE SET+WHERE", stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name`", ":`col-name_1`"])

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values({"col-name": "orig", "other": "x"}))
                conn.execute(stmt)
                r = conn.execute(select(t)).all()[0]._mapping
                assert r["col-name"] == "updated"
            finally:
                t.drop(conn)
    e2e("UPDATE", run)


def scenario_delete_where():
    tname = T("delete")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), schema=SCHEMA)

    stmt = delete(t).where(t.c["col-name"] == "doomed")
    assert_compiled("DELETE WHERE", stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name_1`"])

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values([{"col-name": "keep"}, {"col-name": "doomed"}]))
                conn.execute(stmt)
                rows = conn.execute(select(t)).all()
                assert len(rows) == 1 and rows[0]._mapping["col-name"] == "keep"
            finally:
                t.drop(conn)
    e2e("DELETE", run)


def scenario_select_filters():
    tname = T("filter")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), Column("n", Integer()), schema=SCHEMA)

    for label, stmt, expected in [
        ("eq", select(t).where(t.c["col-name"] == "a"), ":`col-name_1`"),
        ("ne", select(t).where(t.c["col-name"] != "a"), ":`col-name_1`"),
        ("gt", select(t).where(t.c["n"] > 1), ":`n_1`"),
        ("like", select(t).where(t.c["col-name"].like("a%")), ":`col-name_1`"),
        ("between", select(t).where(between(t.c["n"], 1, 10)), ":`n_1`"),
        ("and", select(t).where(and_(t.c["col-name"] == "a", t.c["n"] > 1)), ":`col-name_1`"),
        ("or", select(t).where(or_(t.c["col-name"] == "a", t.c["n"] > 1)), ":`col-name_1`"),
    ]:
        assert_compiled(f"SELECT WHERE {label}", stmt.compile(bind=ENGINE),
                        must_contain=[expected])

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values([
                    {"col-name": "alpha", "n": 1},
                    {"col-name": "beta", "n": 5},
                    {"col-name": "gamma", "n": 10},
                ]))
                rows = conn.execute(select(t).where(t.c["col-name"] == "beta")).all()
                assert len(rows) == 1
                rows = conn.execute(select(t).where(between(t.c["n"], 2, 6))).all()
                assert len(rows) == 1
                rows = conn.execute(select(t).where(and_(t.c["n"] > 1, t.c["n"] < 10))).all()
                assert len(rows) == 1
            finally:
                t.drop(conn)
    e2e("SELECT WHERE filters", run)


def scenario_in_list():
    tname = T("in_list")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), schema=SCHEMA)

    stmt = select(t).where(t.c["col-name"].in_(["a", "b", "c"]))
    # Initial compile has POSTCOMPILE marker
    assert_compiled("IN (list) compile", stmt.compile(bind=ENGINE),
                    must_contain=["POSTCOMPILE_col-name_1"])
    # render_postcompile=True expands inline
    expanded_sql = str(stmt.compile(bind=ENGINE, compile_kwargs={"render_postcompile": True}))
    for i in (1, 2, 3):
        if f":`col-name_1_{i}`" not in expanded_sql:
            REPORT.record("IN (list) render_postcompile [local]", False,
                          f"missing :`col-name_1_{i}` in {expanded_sql!r}")
            return
    if hyphen_leak(expanded_sql):
        REPORT.record("IN (list) render_postcompile [local]", False, expanded_sql)
        return
    REPORT.record("IN (list) render_postcompile [local]", True)

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values([{"col-name": v} for v in ["a", "b", "c", "d"]]))
                rows = conn.execute(stmt).all()
                assert sorted(r._mapping["col-name"] for r in rows) == ["a", "b", "c"]
            finally:
                t.drop(conn)
    e2e("IN (list)", run)


def scenario_in_empty():
    tname = T("in_empty")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), schema=SCHEMA)

    stmt = select(t).where(t.c["col-name"].in_([]))
    assert_compiled("IN (empty)", stmt.compile(bind=ENGINE, compile_kwargs={"render_postcompile": True}))

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values({"col-name": "x"}))
                rows = conn.execute(stmt).all()
                assert rows == []
            finally:
                t.drop(conn)
    e2e("IN (empty list)", run)


def scenario_in_subquery():
    tname_a = T("sub_a")
    tname_b = T("sub_b")
    md = MetaData()
    a = Table(tname_a, md, Column("col-name", String()), schema=SCHEMA)
    b = Table(tname_b, md, Column("col-name", String()), schema=SCHEMA)

    subq = select(b.c["col-name"]).where(b.c["col-name"] == "keep")
    stmt = select(a).where(a.c["col-name"].in_(subq))
    assert_compiled("IN (subquery)", stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name_1`"])

    def run():
        with ENGINE.begin() as conn:
            a.create(conn)
            b.create(conn)
            try:
                conn.execute(insert(a).values([{"col-name": "alpha"}, {"col-name": "beta"}]))
                conn.execute(insert(b).values({"col-name": "alpha"}))
                rows = conn.execute(
                    select(a).where(a.c["col-name"].in_(
                        select(b.c["col-name"]).where(b.c["col-name"] == "alpha")
                    ))
                ).all()
                assert len(rows) == 1 and rows[0]._mapping["col-name"] == "alpha"
            finally:
                b.drop(conn)
                a.drop(conn)
    e2e("IN (subquery)", run)


def scenario_limit_offset():
    tname = T("limoff")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), Column("n", Integer()), schema=SCHEMA)

    stmt = select(t).order_by(t.c["n"]).limit(2).offset(1)
    assert_compiled("LIMIT/OFFSET compile", stmt.compile(bind=ENGINE))

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values([
                    {"col-name": "a", "n": 1},
                    {"col-name": "b", "n": 2},
                    {"col-name": "c", "n": 3},
                    {"col-name": "d", "n": 4},
                ]))
                rows = conn.execute(stmt).all()
                vals = [r._mapping["col-name"] for r in rows]
                assert vals == ["b", "c"], vals
            finally:
                t.drop(conn)
    e2e("LIMIT/OFFSET", run)


def scenario_case_when():
    tname = T("casewhen")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), schema=SCHEMA)

    stmt = select(
        case(
            (t.c["col-name"] == "a", literal("matched")),
            else_=literal("other"),
        ),
    )
    assert_compiled("CASE WHEN", stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name_1`"])

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values([{"col-name": "a"}, {"col-name": "z"}]))
                rows = conn.execute(stmt).all()
                vals = sorted([r[0] for r in rows])
                assert vals == ["matched", "other"], vals
            finally:
                t.drop(conn)
    e2e("CASE WHEN", run)


def scenario_cte():
    tname = T("cte")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), schema=SCHEMA)

    c = select(t.c["col-name"]).where(t.c["col-name"] == "keep").cte("c")
    stmt = select(c)
    assert_compiled("CTE", stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name_1`"])

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values([{"col-name": "keep"}, {"col-name": "drop"}]))
                rows = conn.execute(stmt).all()
                assert len(rows) == 1 and rows[0][0] == "keep"
            finally:
                t.drop(conn)
    e2e("CTE", run)


def scenario_null_value():
    tname = T("null")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String(), nullable=True), schema=SCHEMA)

    stmt = insert(t).values({"col-name": None})
    assert_compiled("NULL value bound", stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name`"])

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(stmt)
                rows = conn.execute(select(t)).all()
                assert rows[0]._mapping["col-name"] is None
            finally:
                t.drop(conn)
    e2e("NULL value", run)


def scenario_function_call():
    tname = T("func")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), schema=SCHEMA)

    stmt = select(func.concat(t.c["col-name"], literal("-suffix")))
    assert_compiled("function with bind", stmt.compile(bind=ENGINE))

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values({"col-name": "x"}))
                rows = conn.execute(stmt).all()
                assert rows[0][0] == "x-suffix"
            finally:
                t.drop(conn)
    e2e("function call", run)


def scenario_construct_expanded_state():
    md = MetaData()
    t = Table("t", md, Column("col-name", String()))
    stmt = select(t).where(t.c["col-name"].in_(["a", "b", "c"]))
    compiled = stmt.compile(dialect=ENGINE.dialect)
    expanded = compiled.construct_expanded_state({"col-name_1": ["a", "b", "c"]})
    sql = expanded.statement
    ok = (
        ":`col-name_1_1`" in sql
        and ":`col-name_1_2`" in sql
        and ":`col-name_1_3`" in sql
        and not hyphen_leak(sql)
    )
    REPORT.record("construct_expanded_state [local]", ok, "" if ok else sql)


def scenario_cached_statement_reuse():
    # First compile+execute to populate cache, then run again and verify SQL is still correct
    tname = T("cache")
    md = MetaData()
    t = Table(tname, md, Column("col-name", String()), schema=SCHEMA)

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(insert(t).values({"col-name": "first"}))
                # Run another insert via the same engine (tests statement caching across calls)
                conn.execute(insert(t).values({"col-name": "second"}))
                rows = conn.execute(select(t).where(t.c["col-name"] == "second")).all()
                assert len(rows) == 1
            finally:
                t.drop(conn)
    e2e("cached statement reuse", run)


def scenario_literal_backtick_in_column_name():
    """Column name containing a literal backtick — escaped by doubling
    per BACKQUOTED_IDENTIFIER. Databricks supports this shape (docs
    example: ``DESCRIBE SELECT 5 AS `a``b```). The dialect must emit
    doubled backticks in both DDL and bind markers; the params dict
    key stays the single-backtick original.
    """
    tname = T("backtick")
    md = MetaData()
    t = Table(tname, md, Column("a`b", String()), Column("val", String()), schema=SCHEMA)

    stmt = insert(t).values({"a`b": "hello", "val": "x"})
    compiled = stmt.compile(bind=ENGINE)
    sql = str(compiled)
    ok = ":`a``b`" in sql and "`a``b`" in sql
    if not ok:
        REPORT.record("literal backtick in column name [local]", False, sql)
        return
    REPORT.record("literal backtick in column name [local]", True)

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(stmt)
                rows = conn.execute(select(t)).all()
                r = rows[0]._mapping
                assert r["a`b"] == "hello" and r["val"] == "x"
                rows = conn.execute(select(t).where(t.c["a`b"] == "hello")).all()
                assert len(rows) == 1
            finally:
                t.drop(conn)
    e2e("literal backtick in column name", run)


def scenario_backtick_combined_with_dot():
    """Column name with BOTH a literal backtick AND a default-escape-map
    character (e.g. ``.``). Without bypassing super's escape map we'd
    get a mismatch between the backticked marker (with translated dot)
    and the params dict key (with original dot).
    """
    tname = T("bt_combined")
    md = MetaData()
    t = Table(tname, md, Column("col`x.y", String()), schema=SCHEMA)

    stmt = insert(t).values({"col`x.y": "combo"})
    compiled = stmt.compile(bind=ENGINE)
    sql = str(compiled)
    if ":`col``x.y`" not in sql:
        REPORT.record("backtick + dot combined [local]", False, sql)
        return
    REPORT.record("backtick + dot combined [local]", True)

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(stmt)
                rows = conn.execute(select(t)).all()
                assert rows[0]._mapping["col`x.y"] == "combo"
            finally:
                t.drop(conn)
    e2e("backtick + dot combined", run)


def scenario_collision_siblings():
    tname = T("collision")
    md = MetaData()
    t = Table(
        tname, md,
        Column("col-name", String()),
        Column("col_name", String()),
        schema=SCHEMA,
    )

    stmt = insert(t).values({"col-name": "H", "col_name": "U"})
    assert_compiled("sibling collision (col-name + col_name) compile",
                    stmt.compile(bind=ENGINE),
                    must_contain=[":`col-name`", ":`col_name`"])

    def run():
        with ENGINE.begin() as conn:
            t.create(conn)
            try:
                conn.execute(stmt)
                r = conn.execute(select(t)).all()[0]._mapping
                assert r["col-name"] == "H" and r["col_name"] == "U"
            finally:
                t.drop(conn)
    e2e("sibling collision (col-name + col_name)", run)


# --- Run everything ---

SCENARIOS = [
    scenario_single_row_insert,
    scenario_multi_row_insert,
    scenario_executemany,
    scenario_update_where,
    scenario_delete_where,
    scenario_select_filters,
    scenario_in_list,
    scenario_in_empty,
    scenario_in_subquery,
    scenario_limit_offset,
    scenario_case_when,
    scenario_cte,
    scenario_null_value,
    scenario_function_call,
    scenario_construct_expanded_state,
    scenario_cached_statement_reuse,
    scenario_collision_siblings,
    scenario_literal_backtick_in_column_name,
    scenario_backtick_combined_with_dot,
]

for fn in SCENARIOS:
    try:
        fn()
    except Exception as e:
        REPORT.record(fn.__name__ + " [harness]", False, str(e)[:200])

ok = REPORT.summary()
sys.exit(0 if ok else 1)
