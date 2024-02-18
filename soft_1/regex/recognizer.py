import re

name = r"([a-zA-Z_][a-zA-Z0-9_]*)"
specifier = r"(?:private|protected|public)?"
pair = fr"(?:{specifier}\s{name})"
pairs = fr"{pair}(?:\s*,\s*{pair})*"
class_pattern = fr"class\s+{name}\s*:\s+{pairs}\s+{{}};"
class_regex = re.compile(class_pattern)

try:
    s = input()
except EOFError:
    s = ""

match = class_regex.match(s)
if match is None:
    exit(1)
