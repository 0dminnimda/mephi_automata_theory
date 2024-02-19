from collections import deque

from ply import lex as _ply_lex


class Lexer:
    def __init__(self) -> None:
        self.ply_lexer: _ply_lex.Lexer = _ply_lex.lex(module=self, optimize=True)
        self.reset("")

    def reset(self, text: str):
        self.ok = True
        self.ply_lexer.input(text)

    def next(self):
        if not self.ok:
            return None
        res = self.ply_lexer.token()
        if not self.ok:
            return None
        return res

    tokens = [
        "NAME",
        "COLON",
        "COMMA",
        "END",
    ]

    t_ignore = " \t"
    t_NAME = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
    t_COLON = r":"
    t_COMMA = r","
    t_END = r"{};"

    # def t_NAME(self, t):
    #     r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
    #     self.names.append(t.value)
    #     return t

    def t_error(self, t):
        self.ok = False
        t.lexer.skip(1)


_lexer = Lexer()


def has_duplicates(x):
    seen = set()
    return any(i in seen or seen.add(i) for i in x)


SPECIFIER = {
    "private",
    "protected",
    "public",
}


def match(string: str) -> tuple[bool, str | None]:
    _lexer.reset(string)
    names = deque()

    tok = _lexer.next()
    if not tok or tok.value != "class":
        return False, None

    tok = _lexer.next()
    if not tok or tok.type != "NAME":
        return False, None
    names.append(tok.value)

    tok = _lexer.next()
    if not tok or tok.type != "COLON":
        return False, None

    while 1:
        tok = saved = _lexer.next()
        skip = False
        if not tok:
            return False, None
        elif tok.value in SPECIFIER:
            tok = _lexer.next()
            if not tok:
                return False, None
            elif tok.type == "NAME":
                names.append(tok.value)
            else:
                names.append(saved.value)
                skip = True
        elif tok.type == "NAME":
            names.append(tok.value)
        else:
            return False, None

        if not skip:
            tok = _lexer.next()
        if not tok:
            return False, None
        elif tok.type == "COMMA":
            pass
        elif tok.type == "END":
            break
        else:
            return False, None

    name, *parents = names
    if has_duplicates(parents):
        return True, name
    return True, None
