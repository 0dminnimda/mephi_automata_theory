from __future__ import annotations

import string
import classes as ast
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Sequence, NamedTuple
from collections import deque
from tnfa import TNFA


State = int
Tag = int
Priority = int
E = TypeVar("E")


@dataclass
class TDFA(Generic[E]):
    """
    Tagged Deterministic Finite Automaton
    """

    alphabet: set[E] = field(repr=False)  # Σ
    tags: set[Tag]  # T
    states: set[State]  # Q
    initial_state: State  # S
    final_states: set[State]  # F
    symbol_transitions: set[SymbolTransition[E]]  # ∆ - transitions on alphabet symbols
    epsilon_transitions: set[
        EpsilonTransition
    ]  # ∆ - optionally tagged ϵ-transitions with priority


def determinization(tnfa: TNFA):
    state = set()
    final_state = set()
    phi = []

    r0 = list(range(len(tnfa.tags)))
    fianl_registers = list(range(len(tnfa.tags), 2*len(tnfa.tags)))
    registers = r0 + fianl_registers

    configurations = epsilon_closure([(tnfa.initial_state, r0, )])
    presedances = presedance(configurations)


def epsilon_reachable(tnfa: TNFA, ordered_eps, states: set[State]) -> set[State]:
    stack = deque(((s, []) for s in states))
    result = set()
    enqueued = set(stack)
    while stack:
        state = stack.pop()
        result.add(state)

        for tag, next_state in ordered_eps.get(state, []):
            if next_state not in enqueued:
                stack.append(next_state)

    return result