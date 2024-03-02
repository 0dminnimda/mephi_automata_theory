from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Sequence


State = int
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
class NFA(Generic[E]):
    states: set[State] = field(default_factory=set)  # Q
    alphabet: set[E] = field(default_factory=set, repr=False)  # Sigma
    transitions: dict[tuple[State, E | None], set[State]] = field(
        default_factory=dict
    )  # delta
    initial_states: set[State] = field(default_factory=set)  # S
    final_states: set[State] = field(default_factory=set)  # F

    def shift_states(self, offset: State):
        self.states = {s + offset for s in self.states}
        self.transitions = {
            (q + offset, a): {q + offset for q in qs}
            for (q, a), qs in self.transitions.items()
        }
        self.initial_states = {s + offset for s in self.initial_states}
        self.final_states = {s + offset for s in self.final_states}

    @classmethod
    def disjoin(cls, nfas: Sequence[NFA[E]]):
        current = 0
        for nfa in nfas:
            current_save = current
            current += max(nfa.states) + 1
            nfa.shift_states(current_save)

    def transitions_into(self, state, symbol):
        return self.transitions.get((state, symbol), set())

    def add_epsilon_transition(self, state1, state2):
        self.transitions[(state1, None)] = self.transitions_into(state1, None) | {state2}
        if state2 in self.final_states:
            self.final_states = self.final_states | {state1}

    def all_transitions(self, states, symbol):
        return union_set(self.transitions_into(q, symbol) for q in states)

    def epsilon_reachable(self, states) -> set[State]:
        epsilon_reachable = states
        while 1:
            states = self.all_transitions(states, None)
            if set.issubset(states, epsilon_reachable):
                break
            epsilon_reachable.update(states)
        return epsilon_reachable

    def run(self, word):
        current_states = self.initial_states
        for symbol in word:
            full = self.epsilon_reachable(current_states)
            current_states = self.all_transitions(full, symbol)
        return not set.isdisjoint(current_states, self.final_states)
