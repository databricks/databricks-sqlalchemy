import re
from datetime import date, datetime, time
from numbers import Number
from uuid import UUID
from sqlalchemy.sql import compiler, sqltypes
import logging

logger = logging.getLogger(__name__)


class DatabricksIdentifierPreparer(compiler.IdentifierPreparer):
    """https://docs.databricks.com/en/sql/language-manual/sql-ref-identifiers.html"""

    legal_characters = re.compile(r"^[A-Z0-9_]+$", re.I)

    def __init__(self, dialect):
        # ``escape_quote`` must match ``initial_quote`` so a literal
        # backtick inside a quoted identifier is doubled (``a``b`` —
        # per the ``BACKQUOTED_IDENTIFIER`` lexer rule in Spark SQL).
        # The default from SQLAlchemy is ``"`` which would escape the
        # wrong character, producing invalid DDL like ``a`b``.
        super().__init__(dialect, initial_quote="`", escape_quote="`")


class DatabricksDDLCompiler(compiler.DDLCompiler):
    def post_create_table(self, table):
        post = [" USING DELTA"]
        if table.comment:
            comment = self.sql_compiler.render_literal_value(
                table.comment, sqltypes.String()
            )
            post.append("COMMENT " + comment)

        post.append("TBLPROPERTIES('delta.feature.allowColumnDefaults' = 'enabled')")
        return "\n".join(post)

    def visit_unique_constraint(self, constraint, **kw):
        logger.warning("Databricks does not support unique constraints")
        pass

    def visit_check_constraint(self, constraint, **kw):
        logger.warning("This dialect does not support check constraints")
        pass

    def visit_identity_column(self, identity, **kw):
        """When configuring an Identity() with Databricks, only the always option is supported.
        All other options are ignored.

        Note: IDENTITY columns must always be defined as BIGINT. An exception will be raised if INT is used.

        https://www.databricks.com/blog/2022/08/08/identity-columns-to-generate-surrogate-keys-are-now-available-in-a-lakehouse-near-you.html
        """
        text = "GENERATED %s AS IDENTITY" % (
            "ALWAYS" if identity.always else "BY DEFAULT",
        )
        return text

    def visit_set_column_comment(self, create, **kw):
        return "ALTER TABLE %s ALTER COLUMN %s COMMENT %s" % (
            self.preparer.format_table(create.element.table),
            self.preparer.format_column(create.element),
            self.sql_compiler.render_literal_value(
                create.element.comment, sqltypes.String()
            ),
        )

    def visit_drop_column_comment(self, create, **kw):
        return "ALTER TABLE %s ALTER COLUMN %s COMMENT ''" % (
            self.preparer.format_table(create.element.table),
            self.preparer.format_column(create.element),
        )

    def get_column_specification(self, column, **kwargs):
        """
        Emit a log message if a user attempts to set autoincrement=True on a column.
        See comments in test_suite.py. We may implement implicit IDENTITY using this
        feature in the future, similar to the Microsoft SQL Server dialect.
        """
        if column is column.table._autoincrement_column or column.autoincrement is True:
            logger.warning(
                "Databricks dialect ignores SQLAlchemy's autoincrement semantics. Use explicit Identity() instead."
            )

        colspec = super().get_column_specification(column, **kwargs)
        if column.comment is not None:
            literal = self.sql_compiler.render_literal_value(
                column.comment, sqltypes.STRINGTYPE
            )
            colspec += " COMMENT " + literal

        return colspec


class DatabricksStatementCompiler(compiler.SQLCompiler):
    """Compiler that wraps every bind parameter marker in backticks.

    Databricks named parameter markers only accept bare identifiers
    (``[A-Za-z_][A-Za-z0-9_]*``) unless backtick-quoted. DataFrame-origin
    column names frequently contain hyphens (``col-with-hyphen``), which
    SQLAlchemy would otherwise render as an invalid marker
    ``:col-with-hyphen`` — the parser splits on ``-`` and reports
    UNBOUND_SQL_PARAMETER.

    Wrapping every marker in backticks (``:`col-with-hyphen```) is valid
    for any identifier the Spark SQL grammar accepts, so we wrap
    unconditionally. The backticks are SQL-side quoting only — the
    parameter's logical name is the text between them, so the params
    dict sent to the driver keeps the original unquoted key.

    Implementation: fix ``bindtemplate`` and ``compilation_bindtemplate``
    on the class. Every bind-render path in SQLAlchemy reads one of
    these two attributes (``bindparam_string``,
    ``_literal_execute_expanding_parameter``, and the insertmanyvalues
    path which this dialect doesn't enable), so fixing them at the
    attribute level covers all paths with no method overrides. We use
    property descriptors with no-op setters because ``SQLCompiler.__init__``
    assigns the default templates from ``BIND_TEMPLATES[paramstyle]``
    during its own init — a plain class attribute would be shadowed by
    that instance assignment. The no-op setter silently discards super's
    assignment so our class-level value is always what gets read.
    """

    _BIND_TEMPLATE = ":`%(name)s`"

    # The no-op setter makes ``SQLCompiler.__init__``'s assignment of the
    # default template a silent no-op so our class-level value is what
    # every render path reads. ``# type: ignore[assignment]`` is required
    # because super declares these as ``str``, and a ``property`` is a
    # different type at the static-analysis level (runtime behavior is
    # unchanged — the descriptor returns ``str`` on access).
    bindtemplate = property(  # type: ignore[assignment]
        lambda self: self._BIND_TEMPLATE, lambda self, _: None
    )
    compilation_bindtemplate = property(  # type: ignore[assignment]
        lambda self: self._BIND_TEMPLATE, lambda self, _: None
    )

    def bindparam_string(self, name, **kw):
        # The template ``:`%(name)s``` assumes ``name`` is safe inside
        # backticks — any literal backtick must be doubled per the
        # ``BACKQUOTED_IDENTIFIER`` lexer rule. The doubling affects only
        # the rendered SQL; the params dict key sent to the driver stays
        # the single-backtick original (the server collapses ``  ->  `
        # when it parses the marker name).
        #
        # When a backtick is present, render the marker ourselves rather
        # than delegating to super. Super would otherwise also apply
        # ``bindname_escape_characters`` translation (``.``->``_``,
        # ``[``->``_``, etc.) AND set ``escaped_from``, which together
        # would propagate into ``escaped_bind_names`` and rewrite the
        # params-dict key. The original dict key uses a single backtick
        # and the un-translated form, so the rewrite would create a
        # mismatch with what the server expects when it parses the
        # backtick-quoted marker name. By owning the rendering here we
        # keep ``escaped_bind_names`` empty for these names and the dict
        # key passes through unchanged.
        if (
            "`" in name
            and not kw.get("escaped_from")
            and not kw.get("post_compile", False)
        ):
            accumulate = kw.get("accumulate_bind_names")
            if accumulate is not None:
                accumulate.add(name)
            visited = kw.get("visited_bindparam")
            if visited is not None:
                visited.append(name)
            return self._BIND_TEMPLATE % {"name": name.replace("`", "``")}
        return super().bindparam_string(name, **kw)

    @staticmethod
    def _split_multivalue_bind_name(bind_name):
        """Split SQLAlchemy's ``<col>_m<idx>`` bind names into (column, idx)."""
        match = re.match(r"^(?P<col>.+)_m(?P<idx>\d+)$", bind_name)
        if not match:
            return None
        return match.group("col"), int(match.group("idx"))

    @staticmethod
    def _value_family(value):
        """Return scalar value family; ``None`` means non-scalar/unsupported."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, Number):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, (bytes, bytearray, memoryview)):
            return "binary"
        if isinstance(value, (date, time, datetime)):
            return "temporal"
        if isinstance(value, UUID):
            return "uuid"
        return None

    @staticmethod
    def _has_custom_bind_expression(type_engine):
        """True if the type (or its impl) customizes bind-expression rendering."""
        type_cls = type(type_engine)
        if (
            getattr(type_cls, "bind_expression", None)
            is not sqltypes.TypeEngine.bind_expression
        ):
            return True

        impl = getattr(type_engine, "impl", None)
        if impl is not None:
            impl_cls = type(impl)
            if (
                getattr(impl_cls, "bind_expression", None)
                is not sqltypes.TypeEngine.bind_expression
            ):
                return True
        return False

    def _build_multi_value_cast_plan(self, insert_stmt):
        """Return {bind_name: cast_sql_type} for multi-row VALUES insert binds.

        Cast only *mixed scalar* multi-row bind groups. This avoids breaking
        complex/custom bind types (e.g. ARRAY/MAP/VARIANT) while still fixing
        Spark inline-table incompatibility for object columns that mix
        primitive families (e.g. INT + STRING).
        """
        if not self.dialect.enable_multirow_insert_casts:
            return {}

        if not getattr(insert_stmt, "_multi_values", None):
            return {}

        grouped_binds = {}
        for bind_name, bind_param in self.binds.items():
            split = self._split_multivalue_bind_name(bind_name)
            if split is None:
                continue
            column_name, _ = split
            grouped_binds.setdefault(column_name, []).append((bind_name, bind_param))

        cast_plan = {}
        for bind_entries in grouped_binds.values():
            families = set()
            has_non_scalar = False
            has_custom_bind_expression = False

            for _, bind_param in bind_entries:
                value_family = self._value_family(getattr(bind_param, "value", None))
                if value_family is None:
                    has_non_scalar = True
                    break
                if value_family != "null":
                    families.add(value_family)

                type_engine = getattr(bind_param, "type", None)
                if type_engine is not None and self._has_custom_bind_expression(
                    type_engine
                ):
                    has_custom_bind_expression = True

            if has_non_scalar or has_custom_bind_expression or len(families) <= 1:
                continue

            for bind_name, bind_param in bind_entries:
                type_engine = getattr(bind_param, "type", None)
                if type_engine is None or isinstance(type_engine, sqltypes.NullType):
                    continue

                dialect_type = type_engine._unwrapped_dialect_impl(self.dialect)
                target_type = self.dialect.type_compiler_instance.process(
                    dialect_type, identifier_preparer=self.preparer
                )
                cast_plan[bind_name] = target_type

        return cast_plan

    def _apply_multi_value_casts(self, sql_text, insert_stmt):
        """Wrap selected ``:`name``` markers with ``CAST(... AS <type>)``."""
        cast_plan = self._build_multi_value_cast_plan(insert_stmt)
        if not cast_plan:
            return sql_text

        rendered = sql_text
        for bind_name, target_type in cast_plan.items():
            marker = self._BIND_TEMPLATE % {"name": bind_name.replace("`", "``")}
            rendered = rendered.replace(marker, f"CAST({marker} AS {target_type})")
        return rendered

    def visit_insert(self, insert_stmt, **kw):
        sql_text = super().visit_insert(insert_stmt, **kw)
        return self._apply_multi_value_casts(sql_text, insert_stmt)

    def limit_clause(self, select, **kw):
        """Identical to the default implementation of SQLCompiler.limit_clause except it writes LIMIT ALL instead of LIMIT -1,
        since Databricks SQL doesn't support the latter.

        https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-qry-select-limit.html
        """
        text = ""
        if select._limit_clause is not None:
            text += "\n LIMIT " + self.process(select._limit_clause, **kw)
        if select._offset_clause is not None:
            if select._limit_clause is None:
                text += "\n LIMIT ALL"
            text += " OFFSET " + self.process(select._offset_clause, **kw)
        return text
