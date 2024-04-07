from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Sequence


State = int
Tag = int
Priority = int
E = TypeVar("E")


def union(it, zero):
    value = zero
    for x in it:
        value = value | x
    return value


def union_set(it):
    return union(it, set())


def union_dict(it):
    return union(it, dict())


@dataclass
class TNFA(Generic[E]):
    """
    Tagged Non-deterministic Finite Automaton
    """

    alphabet: set[E] = field(repr=False)  # Σ
    tags: set[Tag]  # T
    states: set[State]  # Q
    initial_state: State  # S
    final_state: State  # F
    symbol_transitions: dict[
        tuple[State, E], State
    ]  # ∆ - transitions on alphabet symbols
    epsilon_transitions: dict[
        State, set[tuple[Priority, Tag | None, State]]
    ]  # ∆ - optionally tagged ϵ-transitions with priority

    # def shift_states_and_tags(self, state_offset: State, tag_offset: Tag):
    #     self.states = {s + state_offset for s in self.states}
    #     self.symbol_transitions = {
    #         (q + state_offset, a): p + state_offset
    #         for (q, a), p in self.symbol_transitions.items()
    #     }
    #     self.epsilon_transitions = {
    #         q + state_offset: {(i, t + tag_offset, p + state_offset) for (i, t, p) in ts}
    #         for q, ts in self.epsilon_transitions.items()
    #     }
    #     self.initial_states = {s + state_offset for s in self.initial_states}
    #     self.final_states = {s + state_offset for s in self.final_states}

    # @classmethod
    # def disjoin(cls, nfas: Sequence[TNFA[E]]):
    #     current_state = 0
    #     current_tag = 0
    #     for tnfa in nfas:
    #         current_state_save = current_state
    #         current_state += max(tnfa.states) + 1
    #         current_tag_save = current_tag
    #         current_tag += max(tnfa.tags) + 1
    #         tnfa.shift_states_and_tags(current_state_save, current_tag_save)

    def ntags(self):
        return TNFA(
            self.alphabet,
            self.tags,
            self.states,
            self.initial_state,
            self.final_state,
            {},
            {
                q: {(1, -t, p) for (_, t, p) in ts if t is not None}
                for q, ts in self.epsilon_transitions.items()
            },
        )

    def add_state(self):
        state = max(self.states) + 1
        self.states.add(state)
        return state

    # def transitions_into(self, state, symbol):
    #     return self.transitions.get((state, symbol), set())

    # def add_epsilon_transition(self, state1, state2):
    #     self.transitions[(state1, None)] = self.transitions_into(state1, None) | {state2}
    #     if state2 in self.final_states:
    #         self.final_states = self.final_states | {state1}

    # def all_transitions(self, states, symbol):
    #     return union_set(self.transitions_into(q, symbol) for q in states)

    # def epsilon_reachable(self, states) -> set[State]:
    #     epsilon_reachable = states
    #     while 1:
    #         states = self.all_transitions(states, None)
    #         if set.issubset(states, epsilon_reachable):
    #             break
    #         epsilon_reachable.update(states)
    #     return epsilon_reachable

    # def run(self, word):
    #     current_states = self.initial_states
    #     for symbol in word:
    #         full = self.epsilon_reachable(current_states)
    #         current_states = self.all_transitions(full, symbol)
    #     return not set.isdisjoint(current_states, self.final_states)


import classes as ast


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


import string


ALPHABET = set(string.printable)


@dataclass
class Ast2Tnfa(Visitor):
    def visit_Epsilon(self, node: ast.Epsilon, state: State):
        return TNFA(ALPHABET, set(), {state}, state, state, dict(), dict())

    def visit_Symbol(self, node: ast.Symbol, state: State):
        state2 = state + 1
        return TNFA(
            ALPHABET,
            set(),
            {state2, state},
            state2,
            state,
            {(state2, node.value): state},
            dict(),
        )

    def visit_Tag(self, node: ast.Tag, state: State):
        state2 = state + 1
        return TNFA(
            ALPHABET,
            {node.value},
            {state2, state},
            state2,
            state,
            dict(),
            {state2: {(1, node.value, state)}},
        )

    def visit_Concat(self, node: ast.Concat, state: State):
        tnfas = []
        for expr in node.expressions:
            tnfas.append(self.visit(expr, state))
            state = tnfas[-1].initial_state

        assert len(tnfas) > 0, "'concatenation' must have at least one expressions"

        alphabet = union((a.alphabet for a in tnfas), set())
        tags = union((a.tags for a in tnfas), set())
        states = union((a.states for a in tnfas), set())
        initial_state = tnfas[0].initial_state
        final_state = tnfas[-1].final_state
        symbol_transitions = union((a.symbol_transitions for a in tnfas), dict())
        epsilon_transitions = union((a.epsilon_transitions for a in tnfas), dict())

        return TNFA(
            alphabet,
            tags,
            states,
            initial_state,
            final_state,
            symbol_transitions,
            epsilon_transitions,
        )

    def or_with_two(self, lhs: TNFA, rhs: ast.RE):
        lhs_prime = lhs.ntags()
        rhs_prime = self.visit(rhs, lhs.initial_state)
        rhs_prime = self.visit(rhs, lhs_prime.initial_state)

    def visit_Or(self, node: ast.Or, state: State):
        assert len(node.expressions) > 0, "'or' must have at least one expressions"

        prev = self.visit(node.expressions[0], state)
        for expr in node.expressions[1:]:
            prev = self.or_with_two(prev, expr)

        # handle_or_with_two

    def visit_AnyNumberOf(self, node: ast.AnyNumberOf, state: State):
        pass

    def visit_Maybe(self, node: ast.Maybe, state: State):
        pass

    def visit_Repeat(self, node: ast.Repeat, state: State):
        pass

    def visit_NamedGroup(self, node: ast.NamedGroup, state: State):
        pass

    def visit_NamedGroupReference(self, node: ast.NamedGroupReference, state: State):
        pass

    # def visit_
    # def to_nfa(self) -> TNFA:
    #     return TNFA({0}, ALPHABET, dict(), {0}, {0})
