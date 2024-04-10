from __future__ import annotations

import classes as ast
from dataclasses import dataclass, field
from typing import TypeVar, Generic, ClassVar
from collections import deque
from copy import deepcopy
from tnfa import TNFA, OrdMapEpsTrans
from enum import Enum, auto


State = int
Tag = int
Priority = int
Register = int
E = TypeVar("E")


# @dataclass
# class TDFA(Generic[E]):
#     """
#     Tagged Deterministic Finite Automaton
#     """

#     alphabet: set[E] = field(repr=False)  # Σ
#     tags: set[Tag]  # T
#     states: set[State]  # Q
#     initial_state: State  # S
#     final_states: set[State]  # F
#     symbol_transitions: set[SymbolTransition[E]]  # ∆ - transitions on alphabet symbols
#     epsilon_transitions: set[
#         EpsilonTransition
#     ]  # ∆ - optionally tagged ϵ-transitions with priority


class RegVal(Enum):
    NOTHING = auto()
    CURRENT = auto()


@dataclass
class Configuration:
    # registers: list[Reg] = field(default_factory=list)
    registers: dict[Tag, Register] = field(default_factory=dict)
    # transition_tags: dict[Tag, bool] = field(default_factory=dict)
    lookahead_tags: dict[Tag, bool] = field(default_factory=dict)

    def set_lookahead_tag(self, tag: Tag | None):
        if tag is None:
            return

        if tag > 0:
            self.lookahead_tags[tag] = True
        else:
            self.lookahead_tags[-tag] = False

    # def get_lookahead_tag(self, tag: Tag | None):
    #     if tag is None:
    #         return None

    #     if tag > 0:
    #         return self.lookahead_tags[tag]
    #     else:
    #         return self.lookahead_tags[-tag]


DetConfs = dict[State, Configuration]
DetPrecs = list[State]


@dataclass(frozen=True)
class DetState:
    # id: State
    confs: DetConfs
    precs: DetPrecs


@dataclass
class SetOp:
    target: Register
    value: RegVal


@dataclass
class CopyOp:
    target: Register
    source: Register
    do_append: bool = False


RegOp = SetOp | CopyOp
RegOps = list[RegOp]


@dataclass
class DeterminableTNFA(Generic[E]):
    states: set[DetState] = field(default_factory=set)
    initial_state: DetState = field(init=False)
    final_states: set[DetState] = field(default_factory=set)
    transition_function: dict[tuple[DetState, E], tuple[DetState, RegOps]] = field(default_factory=dict)
    final_function: dict[DetState, RegOps] = field(default_factory=dict)

    tnfa: TNFA[E] = field(init=False)
    ordered_eps: OrdMapEpsTrans = field(default_factory=OrdMapEpsTrans)
    confs: DetConfs = field(default_factory=DetConfs)
    precs: DetPrecs = field(default_factory=DetPrecs)
    current_reg: int = 0
    current_state: int = -1

    def get_next_reg(self):
        self.current_reg += 1
        return self.current_reg

    def get_next_state(self):
        self.current_state += 1
        return self.current_state

    @classmethod
    def determine(cls, tnfa: TNFA[E]):
        res = cls()
        res.determinization(tnfa)
        return res

    def determinization(self, tnfa: TNFA[E]):
        self.tnfa = tnfa
        self.ordered_eps = tnfa.get_ordered_mapped_epsilon_transitions()

        # r0 = list(range(len(tnfa.tags)))
        # r0 = dict.fromkeys(tnfa.tags, Reg.CURRENT)
        r0 = {tag: self.get_next_reg() for tag in tnfa.tags}
        # fianl_registers = list(range(len(tnfa.tags), 2*len(tnfa.tags)))
        # registers = r0 + fianl_registers

        self.confs = {tnfa.initial_state: Configuration(r0)}
        self.confs = self.epsilon_closure()
        print("g", *self.confs, "g", sep="\n")
        self.precs = self.precedence()
        self.add_state([])

        # for state in self.states:


    def epsilon_closure(self) -> DetConfs:
        stack = deque(self.confs.items())
        enqueued = set(self.confs.keys())
        result = DetConfs()

        while stack:
            state, conf = stack.pop()

            tag_state_list = self.ordered_eps.get(state, [])
            for tag, next_state in tag_state_list:
                if next_state not in enqueued:
                    next_conf = deepcopy(conf)
                    next_conf.set_lookahead_tag(tag)
                    stack.append((next_state, next_conf))
                    enqueued.add(next_state)

            if not tag_state_list:
                result[state] = conf

        return result

    def precedence(self) -> DetPrecs:
        return [state for state, conf in self.confs.items()]

    def add_state(self, regops: RegOps) -> DetState:
        state = DetState(self.confs, self.precs)
        if state in self.states:
            return state

        result = self.map_to_existing_state(state, regops)
        if result is not None:
            mapped_state, mapped_regops = result
            return mapped_state

        self.states.add(state)
        for tnfa_state, conf in state.confs.items():
            if tnfa_state == self.tnfa.final_state:
                self.final_states.add(state)
                self.final_function[state] = self.calculate_final_regops(conf)
        return state

    # def step_on_symbol(self, symbol: str) -> SimConfs:
    #     result = SimConfs()
    #     for conf in self.confs.items():
    #         state = self.mapped_sym.get(conf[0], dict()).get(symbol)
    #         if state is not None:
    #             result[state] = conf[1]
    #     return result

    def map_to_existing_state(self, state: DetState, regops: RegOps) -> tuple[DetState, RegOps] | None:
        for mapped_state in self.states:
            mapping = self.map_state(state, mapped_state, regops)
            if mapping is not None:
                return mapped_state, mapping
        return None

    def map_state(self, state: DetState, to_state: DetState, regops: RegOps) -> RegOps | None:
        if state.confs.keys() != to_state.confs.keys():
            return None

        if not all(
            conf1.lookahead_tags == conf2.lookahead_tags
            for conf1, conf2 in zip(state.confs.values(), to_state.confs.values())
        ):
            return None

        if state.precs != to_state.precs:
            return None

        reg_to_reg1 = dict[Register, Register]()
        reg_to_reg2 = dict[Register, Register]()

        for conf1, conf2 in zip(state.confs.values(), to_state.confs.values()):
            for tag in self.tnfa.tags:
                # assume every tag is multi-tag
                i = conf1.registers[tag]
                j = conf2.registers[tag]
                m_i = reg_to_reg1.get(i, None)
                m_j = reg_to_reg2.get(j, None)
                if m_i is None and m_j is None:
                    reg_to_reg1[i] = j
                    reg_to_reg2[j] = i
                elif m_i != j or m_j != i:
                    return None

        new_regops = regops[:]
        for i, regop in enumerate(new_regops):
            new_regops[i].target = reg_to_reg1.pop(regop.target)

        for j, i in reg_to_reg1.items():
            if j == i:
                continue
            new_regops.append(CopyOp(i, j))

        return topological_sort(new_regops)

    def calculate_final_regops(self, conf: Configuration) -> RegOps:
        return []
        # for tag, reg in conf.registers.items():
        #     if tag in self.tnfa.named_groups_to_tags:
        #         start, end = self.tnfa.named_groups_to_tags[tag]
        #         for i in range(start, end + 1):
        #             conf.registers[i] = reg


def topological_sort(regops: RegOps) -> RegOps | None:
    indegree = {}

    for regop in regops:
        if isinstance(regop, CopyOp):
            indegree[regop.target] = indegree[regop.source] = 0

    for regop in regops:
        if isinstance(regop, CopyOp):
            indegree[regop.source] += 1

    result = []

    nontrivial_cycle = False
    queue = deque(regops)

    while queue:
        queue_copy = deque()
        something_were_added = False

        for regop in queue:
            if indegree[regop.target] == 0:
                result.append(regop)
                something_were_added = True
                if isinstance(regop, CopyOp):
                    indegree[regop.source] -= 1
            else:
                queue_copy.append(regop)

        if not something_were_added and queue_copy:
            if any(
                regop.target != regop.source
                for regop in queue_copy
                if isinstance(regop, CopyOp)
            ):
                nontrivial_cycle = True
            result.extend(queue_copy)
            break  # only cycles left

        queue = queue_copy

    return result if not nontrivial_cycle else None


# @dataclass
# class TDFA:
