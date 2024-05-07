from dataclasses import dataclass, field, asdict
from classes import RE, Epsilon, SymbolRange, Concat, Or, Repeat, NamedGroup, NamedGroupReference


PIPE = "|"
PERCENT = "%"
ELLIPSIS = "…"
QUESTION_MARK = "?"
OPEN_ROUND_BRACKET = "("
CLOSE_ROUND_BRACKET = ")"
OPEN_CURLY_BRACKET = "{"
CLOSE_CURLY_BRACKET = "}"
OPEN_ANGLE_BRACKET = "<"
CLOSE_ANGLE_BRACKET = ">"
META_CHARS = (
    PIPE
    + PERCENT
    + ELLIPSIS
    + QUESTION_MARK
    + OPEN_ROUND_BRACKET
    + CLOSE_ROUND_BRACKET
    + OPEN_CURLY_BRACKET
    + CLOSE_CURLY_BRACKET
    + OPEN_ANGLE_BRACKET
    + CLOSE_ANGLE_BRACKET
)


def iter_unique(x):
    seen = set()
    for it in x:
        if it in seen:
            continue
        seen.add(it)
        yield it


@dataclass
class Parser:
    string: str = ""
    position: int = 0
    current: RE | None = None

    def peek(self, at: int = 0):
        pos = self.position + at
        if pos >= len(self.string):
            return ""
        return self.string[pos]

    def match(self, string: str):
        if self.string[self.position : self.position + len(string)] == string:
            return True
        return False

    def match_and_consume(self, string: str):
        if self.match(string):
            self.position += len(string)
            return True
        return False

    def consume(self, amount: int = 1):
        self.position += amount
        self.position = min(self.position, len(self.string))

    def match_and_consume_spaces(self):
        while self.string[self.position].isspace():
            self.consume()

    def parse(self, string: str):
        self.string = string
        self.position = 0

        res = self.parse_expr()

        assert res is not None, "Parsing of the regex was unsuccesfull"
        return res

    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        # r1|r2
        # r1|r2|r3|...|rn

        expressions = []
        first = True

        while 1:
            expr = self.parse_concat()
            if expr is None:
                expressions.append(Epsilon())
            else:
                expressions.append(expr)

            if self.match_and_consume(PIPE):
                pass
            elif first:
                return expr
            else:
                break
            first = False

        expressions = tuple(iter_unique(expressions))
        assert len(expressions) >= 1

        if len(expressions) == 1:
            return expressions[0]
        return Or(expressions)

    def parse_concat(self):
        # r1r2
        # r1r2r3...rn

        expressions = []

        while 1:
            expr = self.parse_modyfiers()
            if expr is not None:
                expressions.append(expr)
            else:
                break

        if len(expressions) <= 0:
            return None
        if len(expressions) == 1:
            return expressions[0]
        return Concat(tuple(expressions))

    def parse_modyfiers(self):
        # r?
        # r…
        # r...
        # r+
        # r{number}
        # r{min,}
        # r{,max}
        # r{min,max}
        # r{,}

        expr = self.parse_atom()

        if expr is None:
            return None

        while 1:
            if self.match_and_consume(QUESTION_MARK):
                expr = Repeat(expr, 0, 1)
            elif self.match_and_consume(ELLIPSIS) or self.match_and_consume("..."):
                expr = Repeat(expr, 0, None)
            elif self.match_and_consume("+"):
                expr = Repeat(expr, 1, None)
            elif self.match_and_consume(OPEN_CURLY_BRACKET):
                min, max = self.parse_inner_repeat()
                self.match_and_consume_spaces()
                if not self.match_and_consume(CLOSE_CURLY_BRACKET):
                    raise ValueError(f"Expected '}}' at position {self.position}")
                expr = Repeat(expr, min, max)
            else:
                break

        return expr

    def parse_inner_repeat(self):
        # number
        # min,
        # ,max
        # min,max
        # ,

        self.match_and_consume_spaces()
        if self.match_and_consume(","):
            self.match_and_consume_spaces()
            count = self.parse_number()
            if count is None:
                return 0, None  # ,
            return 0, count  # ,max

        self.match_and_consume_spaces()
        count = self.parse_number()
        if count is None:
            raise ValueError(f"Expected number or comma at position {self.position}")

        self.match_and_consume_spaces()
        if self.match_and_consume(","):
            self.match_and_consume_spaces()
            max = self.parse_number()
            if max is None:
                return count, None  # min,
            return count, max  # min,max

        return count, count  # number

    def parse_atom(self):
        return (
            self.parse_named_group()
            or self.parse_grouped_expression()
            or self.parse_symbol()
            or self.parse_group_reference()
        )

    def parse_named_group(self):
        # (<name>r)

        matches = self.match_and_consume(OPEN_ROUND_BRACKET + OPEN_ANGLE_BRACKET)
        if not matches:
            return None

        name = self.parse_string_until(CLOSE_ANGLE_BRACKET)

        if name is None:
            raise ValueError(f"Expected name at {self.position}")

        if not self.match_and_consume(CLOSE_ANGLE_BRACKET):
            raise ValueError(f"Expected '>' at {self.position}")

        expr = self.parse_expr()

        if expr is None:
            expr = Epsilon()

        if not self.match_and_consume(CLOSE_ROUND_BRACKET):
            raise ValueError(f"Expected ')' at {self.position}")

        return NamedGroup(name, expr)

    def parse_grouped_expression(self):
        # (r)

        if not self.match_and_consume(OPEN_ROUND_BRACKET):
            return None

        expr = self.parse_expr()

        if not self.match_and_consume(CLOSE_ROUND_BRACKET):
            raise ValueError(f"Expected ')' at {self.position}")

        return expr

    def parse_group_reference(self):
        # <name>

        if not self.match_and_consume(OPEN_ANGLE_BRACKET):
            return None

        name = self.parse_string_until(CLOSE_ANGLE_BRACKET)

        if name is None:
            raise ValueError(f"Expected name at {self.position}")

        if not self.match_and_consume(CLOSE_ANGLE_BRACKET):
            raise ValueError(f"Expected '>' at {self.position}")

        return NamedGroupReference(name)

    def parse_symbol(self):
        # symbol
        # %meta_symbol%

        if self.peek() not in META_CHARS:
            c = self.peek()
            result = SymbolRange(c, c)
            self.consume()
            return result

        if self.peek() == PERCENT and self.peek(2) == PERCENT:
            self.consume()
            c = self.peek()
            result = SymbolRange(c, c)
            self.consume(2)
            return result

        return None

    def parse_string_until(self, char: str):
        pos = self.position
        while self.peek() and self.peek() != char:
            self.consume()

        if pos == self.position:
            return None
        return self.string[pos : self.position]

    def parse_number(self):
        pos = self.position
        while self.peek() and self.peek().isdigit():
            self.consume()

        if pos == self.position:
            return None
        return int(self.string[pos : self.position])


_parser = Parser()


def parse(string: str) -> RE:
    return _parser.parse(string)
