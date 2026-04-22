import re
from sqlalchemy.sql import compiler, sqltypes
import logging

logger = logging.getLogger(__name__)


class DatabricksIdentifierPreparer(compiler.IdentifierPreparer):
    """https://docs.databricks.com/en/sql/language-manual/sql-ref-identifiers.html"""

    legal_characters = re.compile(r"^[A-Z0-9_]+$", re.I)

    def __init__(self, dialect):
        super().__init__(dialect, initial_quote="`")


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
    """Render every bind parameter marker wrapped in backticks.

    Databricks named parameter markers accept two forms (per the Spark
    SQL grammar ``SqlBaseParser.g4``): a bare ``IDENTIFIER``
    (``[A-Za-z_][A-Za-z0-9_]*``) or a ``quotedIdentifier`` wrapped in
    backticks. DataFrame-origin column names frequently contain hyphens
    (e.g. ``col-with-hyphen``), which SQLAlchemy would otherwise render
    verbatim as an invalid bare marker ``:col-with-hyphen`` — the parser
    splits on ``-`` and reports ``UNBOUND_SQL_PARAMETER``.

    Backticks are valid for *every* identifier (verified end-to-end
    against a Databricks SQL warehouse), so we wrap unconditionally.
    This mirrors Oracle's ``:"name"`` approach to the same grammar
    constraint (see ``dialects/oracle/cx_oracle.py::OracleCompiler_cx_oracle``).
    The backticks are SQL-side *quoting* only: the parameter's logical
    name is still the text between them, so the params dict passed to
    the driver keeps the original unquoted key. We leave
    ``escaped_bind_names`` untouched, so ``construct_params`` passes
    keys through unchanged.

    Two render paths need covering:

    * **Compile-time rendering** — statement compilation calls
      ``bindparam_string`` via ``self.process(statement)``. Oracle
      overrides this same method (``cx_oracle.py:781``) to quote-wrap
      names, and we do the same here.
    * **IN-clause expansion** — SQLAlchemy's
      ``_literal_execute_expanding_parameter`` builds expanded markers
      (``:col-name_1, :col-name_2, ...``) directly from
      ``self.bindtemplate``, bypassing ``bindparam_string``. This method
      is called from three sites: at execute time
      (``default.py::_execute_context``), during compile time when the
      user passes ``compile_kwargs={'render_postcompile': True}``, and
      from ``construct_expanded_state``. We intercept by overriding the
      method itself rather than swapping ``bindtemplate`` in
      ``__init__``, because the ``render_postcompile=True`` path fires
      inside super's own ``__init__`` — before a subclass ``__init__``
      post-super override would take effect.
    """

    _BACKTICKED_BIND_TEMPLATE = ":`%(name)s`"

    def bindparam_string(self, name, **kw):
        # Fall through to super for the specialized render paths it
        # already handles (POSTCOMPILE placeholder; escape-map translation
        # for chars like '.', '[', ']', etc. that super rewrites before
        # rendering). For those cases super's own rendering is correct;
        # we only intercept the primary path where the name is passed
        # through unmodified into the standard bindtemplate.
        if kw.get("post_compile", False) or kw.get("escaped_from"):
            return super().bindparam_string(name, **kw)

        accumulate = kw.get("accumulate_bind_names")
        if accumulate is not None:
            accumulate.add(name)
        visited = kw.get("visited_bindparam")
        if visited is not None:
            visited.append(name)

        ret = self._BACKTICKED_BIND_TEMPLATE % {"name": name}

        bindparam_type = kw.get("bindparam_type")
        if bindparam_type is not None and self.dialect._bind_typing_render_casts:
            type_impl = bindparam_type._unwrapped_dialect_impl(self.dialect)
            if type_impl.render_bind_cast:
                ret = self.render_bind_cast(bindparam_type, type_impl, ret)
        return ret

    def _literal_execute_expanding_parameter(self, name, parameter, values):
        # Super reads ``self.bindtemplate`` (or ``compilation_bindtemplate``
        # for numeric paramstyles) once into a local variable and uses it to
        # render every expanded marker. Swap both to our backticked template
        # for the duration of the call, then restore, so any later read sees
        # the original values. This covers execute-time expansion, the
        # ``render_postcompile=True`` compile-kwarg path that fires inside
        # super's ``__init__``, and ``construct_expanded_state``.
        saved_bt = getattr(self, "bindtemplate", None)
        saved_cbt = getattr(self, "compilation_bindtemplate", None)
        self.bindtemplate = self._BACKTICKED_BIND_TEMPLATE
        self.compilation_bindtemplate = self._BACKTICKED_BIND_TEMPLATE
        try:
            return super()._literal_execute_expanding_parameter(
                name, parameter, values
            )
        finally:
            if saved_bt is not None:
                self.bindtemplate = saved_bt
            if saved_cbt is not None:
                self.compilation_bindtemplate = saved_cbt

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
