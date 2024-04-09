from __future__ import annotations

import classes as ast
from dataclasses import dataclass, field
from typing import TypeVar, Generic, ClassVar
from collections import deque
from copy import deepcopy
from tnfa import TNFA, OrdMapEpsTrans


State = int
Tag = int
Priority = int
Register = int
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


@dataclass
class Configuration:
    state: State
    registers: list[Register]
    transition_tags: dict[Tag, State | None]
    lookahead_tags: dict[Tag, State | None]
    # id: ClassVar[int] = 0

    # def __post_init__(self):
    #     self.id = Configuration.id
    #     Configuration.id += 1

    # def __hash__(self) -> int:
    #     return hash((self.state, self.id))

    def set_tag(self, tag: Tag | None, value: int):
        if tag is None:
            return

        if tag > 0:
            self.lookahead_tags[tag] = value
        else:
            self.lookahead_tags[-tag] = -1


def determinization(tnfa: TNFA):
    states = set()
    final_states = set()
    phi = []

    ordered_eps = tnfa.get_ordered_mapped_epsilon_transitions()

    r0 = list(range(len(tnfa.tags)))
    fianl_registers = list(range(len(tnfa.tags), 2*len(tnfa.tags)))
    registers = r0 + fianl_registers

    tdfa_state = 0
    configurations = epsilon_closure(ordered_eps, tdfa_state, [Configuration(tnfa.initial_state, r0, dict(), dict())])
    # print("g", *configurations, "g", sep="\n")
    # precedences = precedence(configurations)
    # add_state(states, final_states, fianl_registers, phi, configurations)


def epsilon_closure(ordered_eps: OrdMapEpsTrans, tdfa_state: State, confs: list[Configuration]) -> list[Configuration]:
    stack: deque[Configuration] = deque(confs)
    result: list[Configuration] = []
    enqueued: set[State] = {it.state for it in stack}
    while stack:
        conf = stack.pop()

        tag_state_list = ordered_eps.get(conf.state, [])
        # print(conf, tag_state_list)
        for tag, next_state in tag_state_list:
            if next_state not in enqueued:
                next_conf = deepcopy(conf)
                next_conf.state = next_state
                next_conf.set_tag(tag, tdfa_state)
                stack.append(next_conf)
                enqueued.add(next_state)

        if not tag_state_list:
            result.append(conf)

    return result
