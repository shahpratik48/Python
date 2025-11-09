import unittest

from sql_parser import (
    BinaryExpression,
    Identifier,
    Literal,
    SelectStatement,
    Wildcard,
    parse,
    parse_one,
)


class SQLParserTests(unittest.TestCase):
    def test_parse_simple_select(self) -> None:
        statement = parse_one("SELECT name, age FROM users WHERE age > 30;")
        self.assertIsInstance(statement, SelectStatement)
        self.assertEqual(len(statement.columns), 2)
        self.assertEqual(statement.columns[0].alias, None)
        self.assertEqual(statement.from_[0].name.parts, ("users",))

        where = statement.where
        self.assertIsInstance(where, BinaryExpression)
        self.assertEqual(where.operator, ">")
        self.assertIsInstance(where.left, Identifier)
        self.assertEqual(where.left.parts, ("age",))
        self.assertEqual(where.right, Literal(30))

    def test_parse_wildcard_and_alias(self) -> None:
        statement = parse_one("SELECT u.*, COUNT(*) AS total FROM users u;")
        self.assertEqual(len(statement.columns), 2)

        wildcard = statement.columns[0].expression
        self.assertIsInstance(wildcard, Wildcard)
        self.assertEqual(wildcard.qualifier, ("u",))

        aggregate = statement.columns[1]
        self.assertEqual(aggregate.alias, "total")

    def test_parse_multiple_statements(self) -> None:
        statements = parse(
            "SELECT id FROM accounts LIMIT 5; SELECT DISTINCT status FROM accounts;"
        )
        self.assertEqual(len(statements), 2)
        self.assertTrue(statements[0].limit, "Expected first statement to have a limit")
        self.assertTrue(statements[1].distinct)


if __name__ == "__main__":
    unittest.main()
