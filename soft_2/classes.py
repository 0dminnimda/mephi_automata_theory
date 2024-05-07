from dataclasses import dataclass, field


@dataclass(frozen=True)
class RE:
    pass


@dataclass(frozen=True)
class Epsilon(RE):
    pass


@dataclass(frozen=True)
class SymbolRange(RE):
    start: str
    end: str
    accept: bool = True


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

