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
            return any(
                start <= char <= end
                for start, end in self.ranges
            )
        return all(
            not (start <= char <= end)
            for start, end in self.ranges
        )

    @staticmethod
    def ranges_as_intervals(ranges: Iterable[tuple[str, str]]):
        for start, end in ranges:
            if end >= MAX_UNICODE:
                yield start, end
            else:
                yield start, chr(ord(end) + 1)

    @staticmethod
    def intervals_as_ranges(ranges: Iterable[tuple[str, str]]):
        for start, end in ranges:
            if end >= MAX_UNICODE:
                yield start, end
            else:
                yield start, chr(ord(end) - 1)

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
            result.append((prev, chr(ord(start) - 1)))
            prev = chr(ord(end) + 1)
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

