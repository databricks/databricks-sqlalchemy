# Release History

# 2.0.10 (2026-05-26)

- Fix: Quote bind parameter names with backticks so column names containing characters outside `[A-Za-z0-9_]`, including hyphens, bind correctly (databricks/databricks-sqlalchemy#60 by @msrathore-db)
- Fix: Bind UUID values in canonical hyphenated form for `Column(Uuid)` and UUID comparisons (databricks/databricks-sqlalchemy#63 by @sreekanth-db)

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
