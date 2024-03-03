from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Sequence
from collections import defaultdict


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


@dataclass(frozen=True)
class NamedGroupString:
    name: str


@dataclass
class NFA(Generic[E]):
    states: set[State] = field(default_factory=set)  # Q
    alphabet: set[E] = field(default_factory=set, repr=False)  # Sigma
    transitions: dict[tuple[State, E | None], set[State]] = field(
        default_factory=dict
    )  # delta
    initial_states: set[State] = field(default_factory=set)  # S
    final_states: set[State] = field(default_factory=set)  # F
    named_groups: set[str] = field(default_factory=set)
    group_beginings: dict[int, set[str]] = field(default_factory=dict)
    group_endings: dict[int, set[str]] = field(default_factory=dict)

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

    def add_state(self):
        state = max(self.states) + 1
        self.states.add(state)
        return state

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

    def run_group_capturing(self, word, i, states, last_positions, captures):
        for q in states:
            if q in self.group_endings:
                for group in self.group_endings[q]:
                    last_pos = last_positions.get(group)
                    assert last_pos is not None, f"finishing a '{group}' group capture without starting it"
                    captures[group].append(word[last_pos:i])
                    last_positions[group] = None

            if q in self.group_beginings:
                for group in self.group_beginings[q]:
                    assert last_positions.get(group) is None, f"starting a new '{group}' group capture without finishing the previous one"
                    last_positions[group] = i

    def run(self, word):
        last_positions = dict[str, int | None]()
        captures = defaultdict[str, list[str]](list)

        i = -1
        current_states = self.initial_states
        for i, symbol in enumerate(word):
            full = self.epsilon_reachable(current_states)
            self.run_group_capturing(word, i, full, last_positions, captures)
            current_states = self.all_transitions(full, symbol)
            print(full, "->", current_states)

        i += 1
        full = self.epsilon_reachable(current_states)
        self.run_group_capturing(word, i, full, last_positions, captures)

        print(captures)
        print(last_positions)

        return not set.isdisjoint(current_states, self.final_states)
