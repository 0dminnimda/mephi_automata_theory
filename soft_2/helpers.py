from __future__ import annotations

from collections import deque
from typing import TypeVar, Protocol, Any, Sequence


T = TypeVar("T")


class Comparable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...
    def __le__(self, other: Any, /) -> bool: ...


CmpT = TypeVar("CmpT", bound=Comparable, covariant=True)


def merge_intervals(intervals: list[tuple[CmpT, CmpT]]) -> deque[tuple[CmpT, CmpT]]:
    if not intervals:
        return deque()

    intervals.sort(key=lambda x: x[0])

    stack = deque([intervals[0]])
    for start, end in intervals[1:]:
        lstart, lend = stack.pop()
        if lend < start:
            stack.append((lstart, lend))
            stack.append((start, end))
        else:
            stack.append((lstart, max(lend, end)))
        # or
        # if lstart <= start <= lend:
        #     stack.append((lstart, max(lend, end)))
        # else:
        #     stack.append((lstart, lend))
        #     stack.append((start, end))

    return stack


def split_overlapping_intervals(
    intervals: list[tuple[CmpT, CmpT]]
) -> deque[tuple[CmpT, CmpT]]:
    """
    Will return unique non-overlapping intervals, wich will cover all the previously covered points,
    but will be split at the boudaries of all intervals.
    >>> split_overlapping_intervals([(1, 3), (2, 6), (8, 9), (11, 16), (13, 14), (17, 18), (17, 19), (18, 18)])
    [(1, 2), (2, 3), (3, 6), (8, 9), (11, 13), (13, 14), (14, 16), (17, 18), (18, 19)]
    >>> split_overlapping_intervals([('a', 'e'), ('c', 'e'), ('a', 'c')])
    [('a', 'c'), ('c', 'e'), ('e', 'd')]
    """

    bounds = set()
    for start, end in intervals:
        bounds.add(start)
        bounds.add(end)

    sorted_bounds = sorted(bounds)
    result = deque()
    # filter all possible bounds to get only ones, that are covered by the given intervals
    for start_c, end_c in zip(sorted_bounds, sorted_bounds[1:]):
        included = False
        for start_p, end_p in intervals:
            if end_p < start_c or end_c < start_p:
                continue

            if start_p <= start_c and end_c <= end_p:
                included = True
                break

        if included:
            result.append((start_c, end_c))

    return result


def interval_covers_another(
    base: Sequence[tuple[CmpT, CmpT]], covered: Sequence[tuple[CmpT, CmpT]]
) -> bool:
    # it's ok, for now we'll have O(n*m) algo
    for start_c, end_c in covered:
        if not any(start_o <= start_c and end_c <= end_o for start_o, end_o in base):
            return False
    return True


def iter_unique(x):
    seen = set()
    for it in x:
        if it in seen:
            continue
        seen.add(it)
        yield it


def set_intersection_and_difference(s1: set[T], s2: set[T]) -> tuple[set[T], set[T]]:
    r"""
    Returns intersection s1 ∩ s2 and the difference s1 \ s2.
    SEE: https://stackoverflow.com/a/68165720/12141949
    """
    difference = s1 - s2
    intersection = s1 - difference
    return intersection, difference
