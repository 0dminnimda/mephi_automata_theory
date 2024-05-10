from __future__ import annotations

from dataclasses import dataclass, field
from parser import parse
from tnfa import ast_to_tnfa
from tdfa import tnfa_to_tdfa, SimulatableTDFA, TDFA


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
    def __init__(self, regex: str | SimulatableTDFA):
        if isinstance(regex, str):
            self._simulatable = tnfa_to_tdfa(ast_to_tnfa(parse(regex))).as_simulatable()
        else:
            self._simulatable = regex

    def match(self, s: str) -> Match | None:
        groups = self._simulatable.match_whole_string(s, 0)
        if groups is None:
            return None
        return Match(0, len(s), s, groups)

    def findall(self, s: str) -> list[Match]:
        result = []
        index = 0
        while index < len(s):
            next_index, groups = self._simulatable.match_maximum_length(s, index)
            if groups is not None and index != next_index:
                result.append(Match(index, next_index, s[index:next_index], groups))
                index = next_index
            else:
                index += 1
        return result


# def compile(re: str) -:
