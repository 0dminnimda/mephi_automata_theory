import regex

name = r"([a-zA-Z_][a-zA-Z0-9_]*)"
specifier = r"(?:private|protected|public)?"
pair = fr"(?:{specifier}\s{name})"
pairs = fr"{pair}(?:\s*,\s*{pair})*"
class_pattern = fr"class\s+{name}\s*:\s+{pairs}\s+{{}};"
class_regex = regex.compile(class_pattern)


def match(string: str) -> tuple[bool, str | None]:
    match = class_regex.match(string)
    if match is None:
        return False, None
    name = match.group(1)
    parents = match.captures(2) + match.captures(3)
    if len(parents) != len(set(parents)):
        return True, name
    return True, None
