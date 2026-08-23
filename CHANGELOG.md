# Release History

# Unreleased

- Fix: Cache `get_foreign_keys()` per table per reflection pass, so reflecting a schema no longer issues a redundant `DESCRIBE TABLE EXTENDED` per table on top of the one `get_pk_constraint()` already made (fixes #72)

# 2.0.10 (2026-06-18)

- Fix: Quote bind parameter names containing non-identifier characters (e.g. hyphens, backticks) so columns and parameters with special characters bind correctly (databricks/databricks-sqlalchemy#60 by @msrathore-db)
- Fix: Bind UUID values in canonical hyphenated form (databricks/databricks-sqlalchemy#63 by @sreekanth-db, fixes #50)
- Fix: Support pandas multi-row inserts with mixed-type columns via adaptive CAST, including bind names that contain escape-map characters, with an opt-out gate (databricks/databricks-sqlalchemy#68, #69 by @msrathore-db)
- Fix: Use public `user_agent_entry` connect parameter instead of deprecated `_user_agent_entry` to silence the deprecation warning from `databricks-sql-connector >= 4.0.1` (databricks/databricks-sqlalchemy#64 by @jprakash-db)
- Fix: Promote `Float(precision > 24)` to `DOUBLE` in `CREATE TABLE` to preserve precision (databricks/databricks-sqlalchemy#65 by @jayantsing-db)

# 2.0.9 (2026-02-20)

- Feature: Added `pool_pre_ping` support via `do_ping()` override to detect and recycle dead connections (databricks/databricks-sqlalchemy#54 by @msrathore-db)
- Fix: Pinned poetry version in CI workflows to fix build failures (databricks/databricks-sqlalchemy#54 by @msrathore-db)

# 2.0.8 (2025-09-08)

- Feature: Added support for variant datatype (databricks/databricks-sqlalchemy#42 by @msrathore-db)

# 2.0.7 (2025-06-23)

- Feature: Added support for complex data types such as DatabricksArray and DatabricksMap [Private Preview] (databricks/databricks-sqlalchemy#30 by @jprakash-db)

# 2.0.6 (2025-04-29)

- Relaxed pin for `pyarrow` (databricks/databricks-sqlalchemy#20 by @dhirschfeld)

# 2.0.5 (2025-02-22)

- Added support for double column types (databricks/databricks-sqlalchemy#19 by @up-stevesloan)

# 2.0.4 (2025-01-27)

- All the SQLAlchemy features from `databricks-sql-connector>=4.0.0` have been moved to this `databricks-sqlalchemy` library
- Support for SQLAlchemy v2 dialect is provided