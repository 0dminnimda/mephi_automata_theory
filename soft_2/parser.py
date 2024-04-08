from dataclasses import dataclass, field
from classes import RE, Epsilon, Symbol, Concat, Or, AnyNumberOf, Maybe, Repeat, NamedGroup, NamedGroupReference


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
        # r{number}

        expr = self.parse_atom()

        if expr is None:
            return None

        while 1:
            if self.match_and_consume(QUESTION_MARK):
                expr = Maybe(expr)
            elif self.match_and_consume(ELLIPSIS) or self.match_and_consume("..."):
                expr = AnyNumberOf(expr)
            elif self.match_and_consume(OPEN_CURLY_BRACKET):
                count = self.parse_number()
                if count is None:
                    raise ValueError(f"Expected number at position {self.position}")
                if not self.match_and_consume(CLOSE_CURLY_BRACKET):
                    raise ValueError(f"Expected '}}' at position {self.position}")
                expr = Repeat(expr, count)
            else:
                break

        return expr

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
            result = Symbol(self.peek())
            self.consume()
            return result

        if self.peek() == PERCENT and self.peek(2) == PERCENT:
            self.consume()
            result = Symbol(self.peek())
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


parser = Parser()

if __name__ == "__main__":
    re = parser.parse("b|(a|%?%){2}?")
    print(re)
    nfa = re.to_nfa()
    print(nfa)
    assert nfa.run("")
    assert nfa.run("b")
    assert not nfa.run("a")
    assert not nfa.run("?")
    assert nfa.run("aa")
    assert nfa.run("?a")
    assert nfa.run("a?")
    assert nfa.run("??")
    assert not nfa.run("??a")
    assert not nfa.run("?a?")
