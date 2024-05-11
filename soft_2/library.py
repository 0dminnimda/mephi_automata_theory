from __future__ import annotations

from dataclasses import dataclass, field
from parser import parse
from tnfa import ast_to_tnfa, TNFA
from tdfa import tnfa_to_tdfa, SimulatableTDFA, TDFA
import classes as ast
from typing import Iterable


@dataclass
class Match:
    start: int
    end: int
    string: str
    groups: dict[str, list[str]] = field(repr=False)

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclass
class Pattern:
    def __init__(self, regex: str | ast.RE | TNFA | TDFA | SimulatableTDFA):
        if isinstance(regex, str):
            self._simulatable = tnfa_to_tdfa(ast_to_tnfa(parse(regex))).as_simulatable()
        elif isinstance(regex, ast.RE):
            self._simulatable = tnfa_to_tdfa(ast_to_tnfa(regex)).as_simulatable()
        elif isinstance(regex, TNFA):
            self._simulatable = tnfa_to_tdfa(regex).as_simulatable()
        elif isinstance(regex, TDFA):
            self._simulatable = regex.as_simulatable()
        elif isinstance(regex, SimulatableTDFA):
            self._simulatable = regex
        else:
            raise TypeError(f"Invalid type for regex: {type(regex)}, supported types: str, RE, TNFA, TDFA, SimulatableTDFA")

    def match(self, s: str) -> Match | None:
        groups = self._simulatable.match_whole_string(s, 0)
        if groups is None:
            return None
        return Match(0, len(s), s, groups)

    def findall(self, s: str) -> list[str]:
        result = []
        index = 0
        while index < len(s):
            next_index, groups = self._simulatable.match_maximum_length(s, index, capture=False)
            if groups is not None and index != next_index:
                result.append(s[index:next_index])
                index = next_index
            else:
                index += 1
        return result

    def finditer(self, s: str) -> Iterable[Match]:
        index = 0
        while index < len(s):
            next_index, groups = self._simulatable.match_maximum_length(s, index, capture=True)
            if groups is not None and index != next_index:
                yield Match(index, next_index, s[index:next_index], groups)
                index = next_index
            else:
                index += 1

    def restore_regex_via_k_path(self) -> str | None:
        return self._simulatable.restore_regular_expression_via_k_path()


def compile(re: str) -> Pattern:
    return Pattern(re)


def match(re: str, s: str) -> Match | None:
    return Pattern(re).match(s)


def findall(re: str, s: str) -> list[str]:
    return Pattern(re).findall(s)


def finditer(re: str, s: str) -> Iterable[Match]:
    return Pattern(re).finditer(s)
