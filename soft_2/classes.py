from dataclasses import dataclass, field
import string
# from tnfa import TNFA, union, TransitionTarget


# ALPHABET = set(string.printable)


@dataclass(frozen=True)
class RE:
    pass
    # def to_nfa(self) -> TNFA:
    #     raise NotImplementedError


@dataclass(frozen=True)
class Epsilon(RE):
    pass
    # def to_nfa(self) -> TNFA:
    #     return TNFA({0}, ALPHABET, dict(), {0}, {0})


@dataclass(frozen=True)
class Symbol(RE):
    value: str

    # def to_nfa(self) -> TNFA:
    #     return TNFA({0, 1}, ALPHABET, {(0, self.value): {TransitionTarget(1)}}, {0}, {1})


@dataclass(frozen=True)
class Concat(RE):
    expressions: tuple[RE, ...]

    # def to_nfa(self) -> TNFA:
    #     nfas = [expr.to_nfa() for expr in self.expressions]
    #     TNFA.disjoin(nfas)

    #     alphabet = union((a.alphabet for a in nfas), set())
    #     states = union((a.states for a in nfas), set())
    #     transitions = union((a.transitions for a in nfas), dict())
    #     initial_states = nfas[0].initial_states if nfas else set()
    #     final_states = nfas[-1].final_states if nfas else set()
    #     result = TNFA(states, alphabet, transitions, initial_states, final_states)

    #     for a1, a2 in zip(nfas, nfas[1:]):
    #         middle = result.add_state()
    #         for q1 in a1.final_states:
    #             result.add_epsilon_transition(q1, middle)
    #         for q2 in a2.initial_states:
    #             result.add_epsilon_transition(middle, q2)

    #     return result


@dataclass(frozen=True)
class Or(RE):
    expressions: tuple[RE, ...]

    # def to_nfa(self) -> TNFA:
    #     nfas = [expr.to_nfa() for expr in self.expressions]
    #     return self.on_nfas(nfas)

    # @staticmethod
    # def on_nfas(nfas: list[TNFA]) -> TNFA:
    #     TNFA.disjoin(nfas)

    #     alphabet = union((a.alphabet for a in nfas), set())
    #     states = union((a.states for a in nfas), set())
    #     transitions = union((a.transitions for a in nfas), dict())
    #     initial_states = union((a.initial_states for a in nfas), set())
    #     final_states = union((a.final_states for a in nfas), set())
    #     return TNFA(states, alphabet, transitions, initial_states, final_states)


@dataclass(frozen=True)
class AnyNumberOf(RE):
    expr: RE

    # def to_nfa(self) -> TNFA:
    #     tnfa = self.expr.to_nfa()
    #     result = Or.on_nfas([tnfa, Epsilon().to_nfa()])

    #     middle = result.add_state()
    #     for q1 in tnfa.final_states:
    #         result.add_epsilon_transition(q1, middle)
    #     for q2 in tnfa.initial_states:
    #         result.add_epsilon_transition(middle, q2)

    #     return result


@dataclass(frozen=True)
class Maybe(RE):
    expr: RE

    # def to_nfa(self) -> TNFA:
    #     return Or((self.expr, Epsilon())).to_nfa()


@dataclass(frozen=True)
class Repeat(RE):
    expr: RE
    count: int

    # def to_nfa(self) -> TNFA:
    #     return Concat((self.expr,) * self.count).to_nfa()


@dataclass(frozen=True)
class Tag(RE):
    value: int


@dataclass(frozen=True)
class NamedGroup(RE):
    name: str
    expr: RE

    # def to_nfa(self) -> TNFA:
    #     result = self.expr.to_nfa()

    #     start = result.add_state()
    #     end = result.add_state()

    #     for q1 in result.initial_states:
    #         result.add_epsilon_transition(start, q1)
    #         # result.with_capture_starts
    #     for q2 in result.final_states:
    #         result.add_epsilon_transition(q2, end)

    #     result.initial_states = {start}
    #     result.final_states = {end}

    #     raise NotImplementedError
    #     # mark the iniital and final states to save the position
    #     return self.expr.to_nfa()


@dataclass(frozen=True)
class NamedGroupReference(RE):
    name: str

    # def to_nfa(self) -> TNFA:
    #     raise NotImplementedError
