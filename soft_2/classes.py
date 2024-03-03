from dataclasses import dataclass, field
import string
from nfa import NFA, union, NamedGroupString


ALPHABET = set(string.printable)


@dataclass(frozen=True)
class RE:
    def to_nfa(self) -> NFA:
        raise NotImplementedError


@dataclass(frozen=True)
class Epsilon(RE):
    def to_nfa(self) -> NFA:
        return NFA({0}, ALPHABET, dict(), {0}, {0})


@dataclass(frozen=True)
class Symbol(RE):
    value: str

    def to_nfa(self) -> NFA:
        return NFA({0, 1}, ALPHABET, {(0, self.value): {1}}, {0}, {1})


@dataclass(frozen=True)
class Concat(RE):
    expressions: tuple[RE, ...]

    def to_nfa(self) -> NFA:
        nfas = [expr.to_nfa() for expr in self.expressions]
        NFA.disjoin(nfas)

        alphabet = union((a.alphabet for a in nfas), set())
        states = union((a.states for a in nfas), set())
        transitions = union((a.transitions for a in nfas), dict())
        initial_states = nfas[0].initial_states if nfas else set()
        final_states = nfas[-1].final_states if nfas else set()
        named_groups = union((a.named_groups for a in nfas), set())
        group_beginings = union((a.group_beginings for a in nfas), dict())
        group_endings = union((a.group_endings for a in nfas), dict())
        result = NFA(
            states,
            alphabet,
            transitions,
            initial_states,
            final_states,
            named_groups,
            group_beginings,
            group_endings,
        )

        for a1, a2 in zip(nfas, nfas[1:]):
            middle = result.add_state()
            for q1 in a1.final_states:
                result.add_epsilon_transition(q1, middle)
            for q2 in a2.initial_states:
                result.add_epsilon_transition(middle, q2)

        return result


@dataclass(frozen=True)
class Or(RE):
    expressions: tuple[RE, ...]

    def to_nfa(self) -> NFA:
        nfas = [expr.to_nfa() for expr in self.expressions]
        return self.on_nfas(nfas)

    @staticmethod
    def on_nfas(nfas: list[NFA]) -> NFA:
        NFA.disjoin(nfas)

        alphabet = union((a.alphabet for a in nfas), set())
        states = union((a.states for a in nfas), set())
        transitions = union((a.transitions for a in nfas), dict())
        initial_states = union((a.initial_states for a in nfas), set())
        final_states = union((a.final_states for a in nfas), set())
        named_groups = union((a.named_groups for a in nfas), set())
        group_beginings = union((a.group_beginings for a in nfas), dict())
        group_endings = union((a.group_endings for a in nfas), dict())
        return NFA(states, alphabet, transitions, initial_states, final_states, named_groups, group_beginings, group_endings)


@dataclass(frozen=True)
class AnyNumberOf(RE):
    expr: RE

    def to_nfa(self) -> NFA:
        nfa = self.expr.to_nfa()
        result = Or.on_nfas([nfa, Epsilon().to_nfa()])

        middle = result.add_state()
        for q1 in nfa.final_states:
            result.add_epsilon_transition(q1, middle)
        for q2 in nfa.initial_states:
            result.add_epsilon_transition(middle, q2)

        return result


@dataclass(frozen=True)
class Maybe(RE):
    expr: RE

    def to_nfa(self) -> NFA:
        return Or((self.expr, Epsilon())).to_nfa()


@dataclass(frozen=True)
class Repeat(RE):
    expr: RE
    count: int

    def to_nfa(self) -> NFA:
        return Concat((self.expr,) * self.count).to_nfa()


@dataclass(frozen=True)
class NamedGroup(RE):
    name: str
    expr: RE

    def to_nfa(self) -> NFA:
        nfa = self.expr.to_nfa()
        nfa.named_groups.add(self.name)
        # s1 = nfa.add_state()
        for q in nfa.initial_states:
            if q not in nfa.group_beginings:
                nfa.group_beginings[q] = set()
            nfa.group_beginings[q].add(self.name)
        for q in nfa.final_states:
            if q not in nfa.group_endings:
                nfa.group_endings[q] = set()
            nfa.group_endings[q].add(self.name)
        return nfa


@dataclass(frozen=True)
class NamedGroupReference(RE):
    name: str

    def to_nfa(self) -> NFA:
        return NFA({0, 1}, ALPHABET, {(0, NamedGroupString(self.name)): {1}}, {0}, {1})
