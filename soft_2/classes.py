from __future__ import annotations

from dataclasses import dataclass, field
from helpers import merge_intervals, interval_covers_another
from typing import Iterable


MAX_UNICODE_ORD = 0x10FFFF
MAX_UNICODE = chr(MAX_UNICODE_ORD)
MIN_UNICODE_ORD = 0
MIN_UNICODE = chr(MIN_UNICODE_ORD)


@dataclass(frozen=True)
class RE:
    pass


@dataclass(frozen=True)
class Epsilon(RE):
    pass


@dataclass(frozen=True)
class SymbolRanges(RE):
    ranges: tuple[tuple[str, str], ...]  # (start, end) pairs
    accept: bool = True

    def matches(self, char: str) -> bool:
        if self.accept:
            return any(start <= char <= end for start, end in self.ranges)
        return all(not (start <= char <= end) for start, end in self.ranges)

    @staticmethod
    def next_char(s: str) -> str:
        if s >= MAX_UNICODE:
            return s
        return chr(ord(s) + 1)

    @staticmethod
    def prev_char(s: str) -> str:
        if s >= MAX_UNICODE or s <= MIN_UNICODE:
            return s
        return chr(ord(s) - 1)

    @staticmethod
    def ranges_as_intervals(ranges: Iterable[tuple[str, str]]):
        for start, end in ranges:
            yield start, SymbolRanges.next_char(end)

    @staticmethod
    def intervals_as_ranges(ranges: Iterable[tuple[str, str]]):
        for start, end in ranges:
            yield start, SymbolRanges.prev_char(end)

    def with_minimized_ranges(self) -> SymbolRanges:
        # don't include the end, it simplifies intersecting
        extended_ranges = list(self.ranges_as_intervals(self.ranges))
        extended_ranges = merge_intervals(extended_ranges)
        ranges = tuple(self.intervals_as_ranges(extended_ranges))
        return SymbolRanges(ranges, self.accept)

    def minimized_as_accepting(self) -> SymbolRanges:
        # expects ranes to be minimized
        if self.accept:
            return self
        ranges = sorted(self.ranges, key=lambda x: x[0])
        result = []
        prev = MIN_UNICODE
        for start, end in ranges:
            if start != MIN_UNICODE:
                result.append((prev, self.prev_char(start)))
            prev = self.next_char(end)
        if prev != MAX_UNICODE:
            result.append((prev, MAX_UNICODE))
        return SymbolRanges(tuple(result), True)

    @staticmethod
    def merge(items: list[SymbolRanges]) -> list[SymbolRanges]:
        accepting_ranges = []
        rejecting_ranges = []

        for item in items:
            if item.accept:
                accepting_ranges.extend(item.ranges)
            else:
                rejecting_ranges.extend(item.ranges)

        result = []
        if accepting_ranges:
            accepting = SymbolRanges(tuple(accepting_ranges), True)
            result.append(accepting.with_minimized_ranges())
        if rejecting_ranges:
            rejecting = SymbolRanges(tuple(rejecting_ranges), False)
            result.append(rejecting.with_minimized_ranges())
        return result

    def covers(self, other: SymbolRanges) -> bool:
        return interval_covers_another(self.ranges, other.ranges)


def make_symbol(s: str) -> SymbolRanges:
    return SymbolRanges(((s, s),))


@dataclass(frozen=True)
class Concat(RE):
    expressions: tuple[RE, ...]


@dataclass(frozen=True)
class Or(RE):
    expressions: tuple[RE, ...]


@dataclass(frozen=True)
class Repeat(RE):
    expr: RE
    min: int
    max: int | None
    lazy: bool = False  # greedy or lazy


@dataclass(frozen=True)
class Tag(RE):
    value: int


@dataclass(frozen=True)
class NamedGroup(RE):
    name: str
    expr: RE


@dataclass(frozen=True)
class NamedGroupReference(RE):
    name: str


@dataclass
class Visitor:
    def visit(self, node, *args, **kwargs):
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise NotImplementedError(
                f"visit method for node '{type(node).__name__}' is not implemented"
            )
        return visitor(node, *args, **kwargs)


@dataclass
class AstReverse(Visitor):
    def visit_Epsilon(self, node: Epsilon):
        return node

    def visit_Tag(self, node: Tag):
        return node

    def visit_SymbolRanges(self, node: SymbolRanges):
        return node

    def visit_Concat(self, node: Concat):
        return Concat(tuple(self.visit(it) for it in node.expressions[::-1]))

    def visit_Or(self, node: Or):
        return Or(tuple(self.visit(it) for it in node.expressions))

    def visit_Repeat(self, node: Repeat):
        return Repeat(self.visit(node.expr), node.min, node.max)

    def visit_NamedGroup(self, node: NamedGroup):
        return NamedGroup(node.name, self.visit(node.expr))

    def visit_NamedGroupReference(self, node: NamedGroupReference):
        return node


_ast_reverse = AstReverse()


def reverse_ast(node: RE) -> RE:
    return _ast_reverse.visit(node)
