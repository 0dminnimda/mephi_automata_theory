from __future__ import annotations

import classes as ast
from dataclasses import dataclass, field
from typing import TypeVar, Generic, ClassVar, Any
from collections import deque
from copy import deepcopy
from tnfa import TNFA, OrdMapEpsTrans, MapSymTrans
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
    transition_tags: dict[Tag, bool] = field(default_factory=dict, compare=False, hash=False)
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
    confs: DetConfs
    precs: DetPrecs
    id: State = 0


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
    single_mapped_sym: MapSymTrans = field(default_factory=MapSymTrans)
    ordered_eps: OrdMapEpsTrans = field(default_factory=OrdMapEpsTrans)
    confs: DetConfs = field(default_factory=DetConfs)
    precs: DetPrecs = field(default_factory=DetPrecs)
    registers: set[Register] = field(default_factory=set)
    final_registers: dict[Tag, Register] = field(default_factory=dict)
    current_reg: int = 0

    def get_next_reg(self):
        self.current_reg += 1
        self.registers.add(self.current_reg)
        return self.current_reg

    @classmethod
    def determine(cls, tnfa: TNFA[E]):
        res = cls()
        res.determinization(tnfa)
        return res

    def determinization(self, tnfa: TNFA[E]):
        self.tnfa = tnfa
        self.ordered_eps = tnfa.get_ordered_mapped_epsilon_transitions()
        self.single_mapped_sym = tnfa.get_mapped_symbol_transitions()

        r0 = {tag: self.get_next_reg() for tag in tnfa.tags}
        self.final_registers = {tag: self.get_next_reg() for tag in tnfa.tags}
        self.confs = {tnfa.initial_state: Configuration(r0)}
        self.confs = self.epsilon_closure(self.confs)
        self.precs = self.precedence(self.confs)
        self.initial_state = self.add_state([])
        print(self.initial_state)

        for state in self.states:
            v_map: dict[tuple[Tag, RegVal], Register] = {}

            for symbol in tnfa.alphabet:
                self.confs = self.step_on_symbol(state, symbol)
                self.confs = self.epsilon_closure(self.confs)
                regops = self.get_transition_regops(v_map)
                self.precs = self.precedence(self.confs)
                next_state = self.add_state(regops)
                self.transition_function[(state, symbol)] = (next_state, regops)
                print(symbol, regops, next_state)

        return (
            tnfa.alphabet,
            tnfa.tags,
            self.states,
            self.initial_state,
            self.final_states,
            self.registers,
            self.final_registers,
            self.transition_function,
            self.final_function,
        )

    def epsilon_closure(self, confs: DetConfs) -> DetConfs:
        stack = deque(confs.items())
        enqueued = set(confs.keys())
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

    def step_on_symbol(self, state: DetState, symbol: E) -> DetConfs:
        result = DetConfs()
        for tnfa_state in state.precs:
            conf = state.confs[tnfa_state]
            tnfa_p = self.single_mapped_sym.get((tnfa_state, symbol))
            if tnfa_p is not None:
                result[tnfa_state] = Configuration(conf.registers, conf.lookahead_tags)
        return result

    def precedence(self, confs: DetConfs) -> DetPrecs:
        return list(confs.keys())

    def add_state(self, regops: RegOps) -> DetState:
        state = DetState(self.confs, self.precs)
        if state in self.states:
            return state

        mapped_state = self.map_to_existing_state(state, regops)
        if mapped_state is not None:
            return mapped_state

        self.states.add(state)
        for tnfa_state, conf in state.confs.items():
            if tnfa_state == self.tnfa.final_state:
                self.final_states.add(state)
                self.final_function[state] = self.get_final_regops(conf)
        return state

    def map_to_existing_state(self, state: DetState, regops: RegOps) -> DetState | None:
        for mapped_state in self.states:
            if self.map_state(state, mapped_state, regops):
                return mapped_state
        return None

    def map_state(self, state: DetState, to_state: DetState, regops: RegOps) -> bool:
        if state.confs.keys() != to_state.confs.keys():
            return False

        if not all(
            conf1.lookahead_tags == conf2.lookahead_tags
            for conf1, conf2 in zip(state.confs.values(), to_state.confs.values())
        ):
            return False

        if state.precs != to_state.precs:
            return False

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
                    return False

        new_regops = regops[:]
        for i, regop in enumerate(new_regops):
            new_regops[i].target = reg_to_reg1.pop(regop.target)

        for j, i in reg_to_reg1.items():
            if j == i:
                continue
            new_regops.append(CopyOp(i, j))

        return topological_sort(new_regops)

    def get_final_regops(self, conf: Configuration) -> RegOps:
        result = []
        for tag in self.tnfa.tags:
                lt = conf.lookahead_tags.get(tag)
                if lt is not None:
                    v = self.regop_rhs(conf.registers, lt, tag)
                    i = self.final_registers[tag]
                    result.append(SetOp(i, v))
        return result

    def get_transition_regops(self, v_map: dict[tuple[Tag, RegVal], Register]) -> RegOps:
        result = []

        for conf in self.confs.values():
            for tag in self.tnfa.tags:
                ht = conf.transition_tags.get(tag)
                if ht is not None:
                    v = self.regop_rhs(conf.registers, ht, tag)
                    i = v_map.get((tag, v))
                    if i is None:
                        i = self.get_next_reg()
                        v_map[(tag, v)] = i
                        result.append(SetOp(i, v))
                    conf.registers[tag] = i

        return result

    def regop_rhs(self, registers: dict[Tag, Register], hist: bool, tag: Tag) -> RegVal:
        # assume every tag is multi-tag
        # return (registers[tag], hist)
        # FIXME: IDK what this function does
        if hist:
            return RegVal.CURRENT
        else:
            return RegVal.NOTHING


def topological_sort(regops: RegOps) -> bool:
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

    regops[:] = result
    return nontrivial_cycle


# @dataclass
# class TDFA:
