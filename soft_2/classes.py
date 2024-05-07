from dataclasses import dataclass, field


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

