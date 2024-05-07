from dataclasses import dataclass, field, asdict
from classes import RE, Epsilon, SymbolRange, Concat, Or, Repeat, NamedGroup, NamedGroupReference
from typing import NoReturn


PIPE = "|"
PERCENT = "%"
ELLIPSIS = "…"
PLUS = "+"
QUESTION_MARK = "?"
OPEN_ROUND_BRACKET = "("
CLOSE_ROUND_BRACKET = ")"
OPEN_CURLY_BRACKET = "{"
CLOSE_CURLY_BRACKET = "}"
OPEN_ANGLE_BRACKET = "<"
CLOSE_ANGLE_BRACKET = ">"
OPEN_SQUARE_BRACKET = "["
CLOSE_SQUARE_BRACKET = "]"
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
    + OPEN_SQUARE_BRACKET
    + CLOSE_SQUARE_BRACKET
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

    def report(self, message) -> NoReturn:
        raise ValueError(f"{message} at position {self.position}")

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
            elif self.match_and_consume(PLUS):
                expr = Repeat(expr, 1, None)
            elif self.match_and_consume(OPEN_CURLY_BRACKET):
                min, max = self.parse_inner_repeat()
                self.match_and_consume_spaces()
                if not self.match_and_consume(CLOSE_CURLY_BRACKET):
                    self.report(f"Expected '{CLOSE_CURLY_BRACKET}'")
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
            self.report("Expected number or comma")

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
            or self.parse_symbol_set()
            or self.parse_symbol(META_CHARS)
            or self.parse_group_reference()
        )

    def parse_named_group(self):
        # (<name>r)

        matches = self.match_and_consume(OPEN_ROUND_BRACKET + OPEN_ANGLE_BRACKET)
        if not matches:
            return None

        name = self.parse_string_until(CLOSE_ANGLE_BRACKET)

        if name is None:
            self.report("Expected name")

        if not self.match_and_consume(CLOSE_ANGLE_BRACKET):
            self.report(f"Expected '{CLOSE_ANGLE_BRACKET}'")

        expr = self.parse_expr()

        if expr is None:
            expr = Epsilon()

        if not self.match_and_consume(CLOSE_ROUND_BRACKET):
            self.report(f"Expected '{CLOSE_ROUND_BRACKET}'")

        return NamedGroup(name, expr)

    def parse_grouped_expression(self):
        # (r)

        if not self.match_and_consume(OPEN_ROUND_BRACKET):
            return None

        expr = self.parse_expr()

        if not self.match_and_consume(CLOSE_ROUND_BRACKET):
            self.report(f"Expected '{CLOSE_ROUND_BRACKET}'")

        return expr

    def parse_group_reference(self):
        # <name>

        if not self.match_and_consume(OPEN_ANGLE_BRACKET):
            return None

        name = self.parse_string_until(CLOSE_ANGLE_BRACKET)

        if name is None:
            self.report("Expected name")

        if not self.match_and_consume(CLOSE_ANGLE_BRACKET):
            self.report(f"Expected '{CLOSE_ANGLE_BRACKET}'")

        return NamedGroupReference(name)

    def parse_symbol_set(self):
        # [a]
        # [ab]
        # [abc]
        # ...

        # [a-z]
        # [a-z0-9]
        # ...

        # [%-%]
        # [%]%]
        # [^...]

        # and any mix of them

        if not self.match_and_consume(OPEN_SQUARE_BRACKET):
            return None

        MINUS = "-"
        CARET = "^"
        META = MINUS + CLOSE_SQUARE_BRACKET

        accept = True
        ranges = []

        if self.peek() == CARET:
            self.consume()
            accept = False

        while 1:
            symb1 = self.parse_symbol(META, accept)
            if symb1 is None:
                break

            if self.peek() == MINUS:
                self.consume()

                symb2 = self.parse_symbol(META, accept)
                if symb2 is None:
                    self.report(f"Expected to get the second part of the symbol range")

                ranges.append(SymbolRange(symb1.start, symb2.end, accept))
            else:
                ranges.append(symb1)

        if not self.match_and_consume(CLOSE_SQUARE_BRACKET):
            self.report(f"Expected '{CLOSE_SQUARE_BRACKET}'")

        return Or(tuple(ranges))

    def parse_symbol(self, meta_chars: str, accept: bool = True):
        # symbol
        # %meta_symbol%

        if self.peek() not in META_CHARS:
            c = self.peek()
            result = SymbolRange(c, c, accept)
            self.consume()
            return result

        if self.peek() == PERCENT and self.peek(2) == PERCENT:
            self.consume()
            c = self.peek()
            result = SymbolRange(c, c, accept)
            self.consume(2)
            return result

        return None

    def parse_string_until(self, char: str) -> str | None:
        pos = self.position
        while self.peek() and self.peek() != char:
            self.consume()

        if pos == self.position:
            return None
        return self.string[pos : self.position]

    def parse_number(self) -> int | None:
        pos = self.position
        while self.peek() and self.peek().isdigit():
            self.consume()

        if pos == self.position:
            return None
        return int(self.string[pos : self.position])


_parser = Parser()


def parse(string: str) -> RE:
    return _parser.parse(string)
