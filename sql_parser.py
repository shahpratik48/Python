"""
Lightweight SQL parser implemented in pure Python.

This module tokenises and parses a practical subset of ANSI SQL focused on
``SELECT`` statements.  It produces a structured, dataclass-based abstract
syntax tree (AST) that is easy to inspect, transform, or serialise.

Supported features
------------------
* ``SELECT`` lists with expressions, aliases, and ``DISTINCT``
* ``FROM`` clause with comma separated table references and aliases
* ``WHERE`` boolean expressions with the operators ``AND``, ``OR``, ``NOT``
  together with comparison and arithmetic operators
* ``GROUP BY`` with optional ``HAVING``
* ``ORDER BY`` with ``ASC`` / ``DESC``
* ``LIMIT`` and ``OFFSET``
* Function calls (e.g. ``COUNT(col)``) and wildcard selections (``*`` or
  ``table.*``)

The grammar is intentionally small but extensible.  To add more SQL constructs,
extend the token definitions or parsing functions following the existing
patterns.  No third-party dependencies are required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union


class SQLParserError(ValueError):
    """Raised when the SQL parser encounters an unrecoverable error."""


@dataclass(frozen=True)
class Token:
    type: str
    value: str
    position: int
    line: int
    column: int


class Expression:
    """Base class for all expression nodes."""


@dataclass(frozen=True)
class Identifier(Expression):
    parts: Tuple[str, ...]

    def __str__(self) -> str:
        return ".".join(self.parts)


@dataclass(frozen=True)
class Literal(Expression):
    value: Union[str, int, float, bool, None]


@dataclass(frozen=True)
class UnaryExpression(Expression):
    operator: str
    operand: Expression


@dataclass(frozen=True)
class BinaryExpression(Expression):
    left: Expression
    operator: str
    right: Expression


@dataclass(frozen=True)
class FunctionCall(Expression):
    name: Identifier
    arguments: Tuple[Expression, ...]
    distinct: bool = False


@dataclass(frozen=True)
class Wildcard(Expression):
    qualifier: Optional[Tuple[str, ...]] = None

    def __str__(self) -> str:
        if self.qualifier:
            return ".".join(self.qualifier) + ".*"
        return "*"


@dataclass(frozen=True)
class OrderByItem:
    expression: Expression
    direction: str  # 'ASC' or 'DESC'


@dataclass(frozen=True)
class TableReference:
    name: Identifier
    alias: Optional[str] = None


@dataclass(frozen=True)
class SelectItem:
    expression: Expression
    alias: Optional[str] = None


@dataclass(frozen=True)
class SelectStatement:
    distinct: bool
    columns: Tuple[SelectItem, ...]
    from_: Tuple[TableReference, ...]
    where: Optional[Expression]
    group_by: Tuple[Expression, ...]
    having: Optional[Expression]
    order_by: Tuple[OrderByItem, ...]
    limit: Optional[int]
    offset: Optional[int]

    def as_dict(self) -> dict:
        """Serialise the statement to a basic Python dictionary."""
        return {
            "type": "select",
            "distinct": self.distinct,
            "columns": [
                {
                    "expression": _expression_to_repr(item.expression),
                    "alias": item.alias,
                }
                for item in self.columns
            ],
            "from": [
                {
                    "name": ".".join(ref.name.parts),
                    "alias": ref.alias,
                }
                for ref in self.from_
            ],
            "where": _expression_to_repr(self.where),
            "group_by": [_expression_to_repr(expr) for expr in self.group_by],
            "having": _expression_to_repr(self.having),
            "order_by": [
                {
                    "expression": _expression_to_repr(item.expression),
                    "direction": item.direction,
                }
                for item in self.order_by
            ],
            "limit": self.limit,
            "offset": self.offset,
        }


Statement = SelectStatement


KEYWORDS = {
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "AS",
    "GROUP",
    "BY",
    "HAVING",
    "ORDER",
    "ASC",
    "DESC",
    "LIMIT",
    "OFFSET",
    "DISTINCT",
    "TRUE",
    "FALSE",
    "NULL",
    "ON",
    "IS",
    "LIKE",
}

TOKEN_REGEX = re.compile(
    r"""
    (?P<SPACE>\s+)
  | (?P<COMMENT>--[^\n]*)
  | (?P<MULTILINE_COMMENT>/\*.*?\*/)
  | (?P<NUMBER>\d+(?:\.\d+)?)
  | (?P<STRING>'([^']|'')*')
  | (?P<IDENTIFIER>[A-Za-z_][\w$]*)
  | (?P<OPERATOR><=|>=|<>|!=|=|<|>)
  | (?P<OTHER>[,()*+/\-.;\.])
    """,
    re.VERBOSE | re.DOTALL,
)

SINGLE_CHAR_TOKENS = {
    ",": "COMMA",
    "(": "LPAREN",
    ")": "RPAREN",
    "*": "STAR",
    "+": "PLUS",
    "-": "MINUS",
    "/": "SLASH",
    ";": "SEMICOLON",
    ".": "DOT",
}

BINARY_PRECEDENCE = {
    "OR": 1,
    "AND": 2,
    "=": 3,
    "!=": 3,
    "<>": 3,
    "<": 3,
    "<=": 3,
    ">": 3,
    ">=": 3,
    "LIKE": 3,
    "IS": 3,
    "+": 4,
    "-": 4,
    "*": 5,
    "/": 5,
}

RIGHT_ASSOCIATIVE = {"IS"}


def tokenize(sql: str) -> List[Token]:
    """Convert an SQL string into a list of tokens."""
    tokens: List[Token] = []
    pos = 0
    line = 1
    column = 1
    length = len(sql)

    while pos < length:
        match = TOKEN_REGEX.match(sql, pos)
        if not match:
            snippet = sql[pos : min(pos + 20, length)]
            raise SQLParserError(
                f"Unexpected character at line {line}, column {column}: {snippet!r}"
            )

        kind = match.lastgroup or ""
        text = match.group()
        start_pos = pos
        pos = match.end()

        if kind in {"SPACE", "COMMENT", "MULTILINE_COMMENT"}:
            line, column = _advance_position(text, line, column)
            continue

        token_line, token_column = line, column
        line, column = _advance_position(text, line, column)

        if kind == "NUMBER":
            token_value = text
            token_type = "NUMBER"
        elif kind == "STRING":
            token_value = text[1:-1].replace("''", "'")
            token_type = "STRING"
        elif kind == "IDENTIFIER":
            upper = text.upper()
            if upper in KEYWORDS:
                token_type = upper
                token_value = upper
            else:
                token_type = "IDENTIFIER"
                token_value = text
        elif kind == "OPERATOR":
            token_type = "OPERATOR"
            token_value = text.upper()
        elif kind == "OTHER":
            token_type = SINGLE_CHAR_TOKENS[text]
            token_value = text
        else:
            raise SQLParserError(f"Unhandled token type: {kind}")

        tokens.append(
            Token(
                type=token_type,
                value=token_value,
                position=start_pos,
                line=token_line,
                column=token_column,
            )
        )

    tokens.append(Token("EOF", "", length, line, column))
    return tokens


class Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = tokens
        self.index = 0

    def peek(self, offset: int = 0) -> Token:
        idx = self.index + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[idx]

    def advance(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def match(self, *types: str) -> Optional[Token]:
        if self.peek().type in types:
            return self.advance()
        return None

    def expect(self, *types: str) -> Token:
        token = self.peek()
        if token.type not in types:
            expected = ", ".join(types)
            raise SQLParserError(
                f"Expected {expected} at line {token.line}, column {token.column}, "
                f"found {token.type}"
            )
        return self.advance()

    def parse(self) -> List[Statement]:
        statements: List[Statement] = []
        while self.peek().type != "EOF":
            if self.match("SEMICOLON"):
                continue
            statements.append(self.parse_statement())
            self.match("SEMICOLON")
        return statements

    def parse_statement(self) -> Statement:
        if self.peek().type == "SELECT":
            return self.parse_select()
        token = self.peek()
        raise SQLParserError(
            f"Unsupported statement starting with {token.type} "
            f"at line {token.line}, column {token.column}"
        )

    def parse_select(self) -> SelectStatement:
        self.expect("SELECT")
        distinct = bool(self.match("DISTINCT"))

        columns = self.parse_select_list()
        from_clause = self.parse_from_clause()
        where_clause = self.parse_where_clause()
        group_by = self.parse_group_by_clause()
        having = self.parse_having_clause()
        order_by = self.parse_order_by_clause()
        limit = self.parse_limit_clause()
        offset = self.parse_offset_clause()

        return SelectStatement(
            distinct=distinct,
            columns=columns,
            from_=from_clause,
            where=where_clause,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )

    def parse_select_list(self) -> Tuple[SelectItem, ...]:
        items: List[SelectItem] = []
        while True:
            items.append(self.parse_select_item())
            if not self.match("COMMA"):
                break
        return tuple(items)

    def parse_select_item(self) -> SelectItem:
        if self.peek().type == "STAR":
            self.advance()
            expression: Expression = Wildcard()
        elif (
            self.peek().type == "IDENTIFIER"
            and self.peek(1).type == "DOT"
            and self.peek(2).type in {"STAR", "IDENTIFIER"}
        ):
            identifier, wildcard = self.parse_identifier(allow_wildcard=True)
            if wildcard:
                qualifier = identifier[:-1]
                expression = Wildcard(qualifier=qualifier)
            else:
                expression = Identifier(identifier)
        else:
            expression = self.parse_expression()

        alias: Optional[str] = None
        if self.match("AS"):
            alias_token = self.expect("IDENTIFIER")
            alias = alias_token.value
        elif self.peek().type == "IDENTIFIER":
            alias = self.advance().value

        return SelectItem(expression=expression, alias=alias)

    def parse_from_clause(self) -> Tuple[TableReference, ...]:
        tables: List[TableReference] = []
        if not self.match("FROM"):
            return tuple()

        while True:
            name_parts, _ = self.parse_identifier()
            alias = None
            if self.match("AS"):
                alias = self.expect("IDENTIFIER").value
            elif self.peek().type == "IDENTIFIER":
                alias = self.advance().value

            tables.append(
                TableReference(
                    name=Identifier(name_parts),
                    alias=alias,
                )
            )

            if not self.match("COMMA"):
                break

        return tuple(tables)

    def parse_where_clause(self) -> Optional[Expression]:
        if not self.match("WHERE"):
            return None
        return self.parse_expression()

    def parse_group_by_clause(self) -> Tuple[Expression, ...]:
        if not self.match("GROUP"):
            return tuple()
        self.expect("BY")
        expressions: List[Expression] = []
        while True:
            expressions.append(self.parse_expression())
            if not self.match("COMMA"):
                break
        return tuple(expressions)

    def parse_having_clause(self) -> Optional[Expression]:
        if not self.match("HAVING"):
            return None
        return self.parse_expression()

    def parse_order_by_clause(self) -> Tuple[OrderByItem, ...]:
        if not self.match("ORDER"):
            return tuple()
        self.expect("BY")
        items: List[OrderByItem] = []
        while True:
            expr = self.parse_expression()
            direction = "ASC"
            if self.match("ASC"):
                direction = "ASC"
            elif self.match("DESC"):
                direction = "DESC"
            items.append(OrderByItem(expr, direction))
            if not self.match("COMMA"):
                break
        return tuple(items)

    def parse_limit_clause(self) -> Optional[int]:
        if not self.match("LIMIT"):
            return None
        value_token = self.expect("NUMBER")
        return _coerce_number(value_token.value)

    def parse_offset_clause(self) -> Optional[int]:
        if not self.match("OFFSET"):
            return None
        value_token = self.expect("NUMBER")
        return _coerce_number(value_token.value)

    def parse_expression(self, min_precedence: int = 1) -> Expression:
        expr = self.parse_unary_expression()

        while True:
            operator = self._current_binary_operator()
            if operator is None:
                break
            precedence = BINARY_PRECEDENCE.get(operator)
            if precedence is None or precedence < min_precedence:
                break

            self.advance()
            next_min = precedence + (0 if operator in RIGHT_ASSOCIATIVE else 1)
            rhs = self.parse_expression(next_min)
            expr = BinaryExpression(expr, operator, rhs)

        return expr

    def _current_binary_operator(self) -> Optional[str]:
        token = self.peek()
        if token.type in {"AND", "OR", "LIKE", "IS"}:
            return token.value
        if token.type == "OPERATOR":
            return token.value
        if token.type in {"PLUS", "MINUS", "STAR", "SLASH"}:
            return token.value
        return None

    def parse_unary_expression(self) -> Expression:
        token = self.peek()
        if token.type in {"NOT", "PLUS", "MINUS"}:
            operator = token.value
            self.advance()
            operand = self.parse_unary_expression()
            return UnaryExpression(operator, operand)
        return self.parse_primary()

    def parse_primary(self) -> Expression:
        token = self.peek()
        if token.type == "LPAREN":
            self.advance()
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr
        if token.type == "NUMBER":
            self.advance()
            return Literal(_coerce_literal_number(token.value))
        if token.type == "STRING":
            self.advance()
            return Literal(token.value)
        if token.type in {"TRUE", "FALSE"}:
            self.advance()
            return Literal(token.type == "TRUE")
        if token.type == "NULL":
            self.advance()
            return Literal(None)
        if token.type == "IDENTIFIER":
            parts, _ = self.parse_identifier()
            identifier = Identifier(parts)
            if self.match("LPAREN"):
                arguments: List[Expression] = []
                distinct = bool(self.match("DISTINCT"))
                if self.peek().type != "RPAREN":
                    while True:
                        arguments.append(self.parse_expression())
                        if not self.match("COMMA"):
                            break
                self.expect("RPAREN")
                return FunctionCall(identifier, tuple(arguments), distinct=distinct)
            return identifier
        if token.type == "STAR":
            self.advance()
            return Wildcard()

        raise SQLParserError(
            f"Unexpected token {token.type} at line {token.line}, column {token.column}"
        )

    def parse_identifier(
        self, allow_wildcard: bool = False
    ) -> Tuple[Tuple[str, ...], bool]:
        token = self.expect("IDENTIFIER")
        parts: List[str] = [token.value]
        wildcard = False

        while self.peek().type == "DOT":
            self.advance()
            if allow_wildcard and self.peek().type == "STAR":
                self.advance()
                parts.append("*")
                wildcard = True
                break
            next_token = self.expect("IDENTIFIER")
            parts.append(next_token.value)

        return tuple(parts), wildcard


def parse(sql: str) -> List[Statement]:
    """
    Parse the supplied SQL string into a list of statements.

    Parameters
    ----------
    sql:
        SQL text containing one or more ``SELECT`` statements.  Statements can be
        separated by semicolons.

    Returns
    -------
    list[Statement]
        A list of parsed statement objects.
    """
    tokens = tokenize(sql)
    parser = Parser(tokens)
    return parser.parse()


def parse_one(sql: str) -> Statement:
    """
    Parse a single SQL statement and return the resulting AST node.

    Raises
    ------
    SQLParserError
        If the SQL contains zero or more than one statement, or if the statement
        type is unsupported.
    """
    statements = parse(sql)
    if not statements:
        raise SQLParserError("No SQL statement found")
    if len(statements) > 1:
        raise SQLParserError("Expected a single statement")
    return statements[0]


def _advance_position(text: str, line: int, column: int) -> Tuple[int, int]:
    line_breaks = text.count("\n")
    if line_breaks:
        line += line_breaks
        column = len(text) - text.rfind("\n")
    else:
        column += len(text)
    return line, column


def _coerce_number(value: str) -> int:
    number = _coerce_literal_number(value)
    if isinstance(number, float):
        if not number.is_integer():
            raise SQLParserError(f"Expected integer literal, found {value!r}")
        return int(number)
    return number


def _coerce_literal_number(value: str) -> Union[int, float]:
    if "." in value:
        return float(value)
    return int(value)


def _expression_to_repr(expr: Optional[Expression]) -> Optional[Union[str, dict]]:
    if expr is None:
        return None
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, Identifier):
        return ".".join(expr.parts)
    if isinstance(expr, Wildcard):
        return str(expr)
    if isinstance(expr, UnaryExpression):
        return {
            "type": "unary",
            "operator": expr.operator,
            "operand": _expression_to_repr(expr.operand),
        }
    if isinstance(expr, BinaryExpression):
        return {
            "type": "binary",
            "operator": expr.operator,
            "left": _expression_to_repr(expr.left),
            "right": _expression_to_repr(expr.right),
        }
    if isinstance(expr, FunctionCall):
        return {
            "type": "function",
            "name": ".".join(expr.name.parts),
            "arguments": [_expression_to_repr(arg) for arg in expr.arguments],
            "distinct": expr.distinct,
        }
    return {"type": expr.__class__.__name__}


def _format_statement(stmt: Statement) -> str:
    """Return a formatted string representation of a statement."""
    return f"Select(distinct={stmt.distinct}, columns={len(stmt.columns)})"


def format_statements(statements: Iterable[Statement]) -> str:
    """Format multiple statements into a human-readable string."""
    return "\n".join(_format_statement(stmt) for stmt in statements)


if __name__ == "__main__":
    example_sql = """
        SELECT DISTINCT customer_id, SUM(total) AS total_spent
        FROM orders
        WHERE status = 'completed' AND total > 50
        GROUP BY customer_id
        HAVING SUM(total) > 100
        ORDER BY total_spent DESC
        LIMIT 10 OFFSET 5;
    """

    parsed = parse(example_sql)
    for statement in parsed:
        print(statement.as_dict())
