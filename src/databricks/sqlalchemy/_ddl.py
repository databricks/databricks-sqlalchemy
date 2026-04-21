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
    # Names that a bare Databricks named-parameter marker (`:name`) accepts:
    # a letter or underscore followed by letters, digits, or underscores.
    # Anything outside that set — hyphens, spaces, dots, brackets, a leading
    # digit, etc. — must be wrapped in backticks (`:`name``), which the
    # Spark/Databricks SQL grammar accepts as a quoted parameter identifier.
    _bindname_is_bare_identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def bindparam_string(self, name, **kw):
        """Render a bind parameter marker.

        Databricks named parameter markers only accept bare identifiers
        ([A-Za-z_][A-Za-z0-9_]*) out of the box. DataFrame-origin column
        names frequently contain hyphens (e.g. ``col-with-hyphen``), which
        SQLAlchemy would otherwise pass through verbatim and produce an
        invalid marker ``:col-with-hyphen`` — the parser splits on ``-``
        and reports UNBOUND_SQL_PARAMETER.

        The Spark SQL grammar accepts a quoted form ``:`col-with-hyphen```,
        mirroring Oracle's ``:"name"`` pattern. The backticks are *quoting*
        only: the parameter's logical name is still the text between them,
        so the params dict sent to the driver must keep the original
        unquoted key. We therefore emit the backticked marker directly
        without populating ``escaped_bind_names`` — leaving the key
        translation in ``construct_params`` a no-op.

        For bare identifiers (the common case), we fall through to the
        default implementation so INSERT/SELECT output stays unchanged.
        """
        if (
            not kw.get("escaped_from")
            and not kw.get("post_compile", False)
            and not self._bindname_is_bare_identifier.match(name)
        ):
            accumulate = kw.get("accumulate_bind_names")
            if accumulate is not None:
                accumulate.add(name)
            visited = kw.get("visited_bindparam")
            if visited is not None:
                visited.append(name)
            quoted = f"`{name}`"
            if self.state is compiler.CompilerState.COMPILING:
                return self.compilation_bindtemplate % {"name": quoted}
            return self.bindtemplate % {"name": quoted}
        return super().bindparam_string(name, **kw)

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
