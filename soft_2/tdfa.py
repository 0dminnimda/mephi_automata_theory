from __future__ import annotations

import classes as ast
from dataclasses import dataclass, field
from typing import TypeVar, Generic, ClassVar, Any
from collections import deque
from copy import deepcopy
from tnfa import TNFA, OrdMapEpsTrans, MapSymTrans, Tag, Priority
from enum import Enum, auto
import tnfa


Register = int
E = TypeVar("E")


class RegVal(Enum):
    NOTHING = auto()
    CURRENT = auto()


@dataclass
class Configuration:
    registers: dict[Tag, Register] = field(default_factory=dict)
    transition_tags: dict[Tag, bool] = field(
        default_factory=dict, compare=False
    )
    lookahead_tags: dict[Tag, bool] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                tuple(self.registers.items()),
                # transition_tags don't matter for state mapping
                tuple(self.lookahead_tags.items()),
            )
        )

    def set_lookahead_tag(self, tag: Tag | None):
        if tag is None:
            return

        if tag > 0:
            self.lookahead_tags[tag] = True
        else:
            self.lookahead_tags[-tag] = False


def confs_as_table(confs) -> str:
    result = f"| state | tags registers | lookahead |\n"
    for s, conf in confs.items():
        result += f"| {s} | {conf.registers} | {conf.lookahead_tags} |\n"
    return result


DetConfs = dict[tnfa.State, Configuration]
DetPrecs = list[tnfa.State]


@dataclass
class DetState:
    id: int = field(compare=False)
    confs: DetConfs
    precs: DetPrecs

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                # id don't matter for state mapping
                tuple(self.confs.items()),
                tuple(self.precs),
            )
        )

    def as_table(self) -> str:
        first = result = f"\n| TDFA state {self.id} |\n"
        result += confs_as_table(self.confs)
        result += "-"*len(first) + "\n"
        return result


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
    states: list[DetState] = field(default_factory=list)
    states_set: set[DetState] = field(default_factory=set)
    initial_state: DetState = field(init=False)
    final_states: set[DetState] = field(default_factory=set)
    transition_function: dict[tuple[DetState, E], tuple[DetState, RegOps]] = field(
        default_factory=dict
    )
    final_function: dict[DetState, RegOps] = field(default_factory=dict)

    tnfa: TNFA[E] = field(init=False)
    single_mapped_sym: MapSymTrans[E] = field(default_factory=MapSymTrans[E])
    ordered_eps: OrdMapEpsTrans = field(default_factory=OrdMapEpsTrans)
    confs: DetConfs = field(default_factory=DetConfs)
    precs: DetPrecs = field(default_factory=DetPrecs)
    registers: set[Register] = field(default_factory=set)
    final_registers: dict[Tag, Register] = field(default_factory=dict)
    current_reg: int = 0
    current_state: int = -1

    def get_next_reg(self):
        self.current_reg += 1
        self.registers.add(self.current_reg)
        return self.current_reg

    def get_next_state(self):
        self.current_state += 1
        self.registers.add(self.current_state)
        return self.current_state

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
        print(self.initial_state.as_table())

        # s (1a2)∗3(a|4b)5b
        # {'g1': (3, 4), 'g2': (1, 2)},

        for state in self.states:
            v_map: dict[tuple[Tag, RegVal], Register] = {}

            for symbol in tnfa.alphabet:
                c1 = self.confs = self.step_on_symbol(state, symbol)
                self.confs = self.epsilon_closure(self.confs)
                if not len(self.confs):
                    continue
                regops = self.get_transition_regops(v_map)
                self.precs = self.precedence(self.confs)
                next_state = self.add_state(regops)
                self.transition_function[(state, symbol)] = (next_state, regops)
                print(symbol, regops, "\n")
                print(symbol, confs_as_table(c1))
                print(next_state.as_table())
                # print(symbol, next_state)
                print()
            exit()

        # for state in self.states:
        #     state.id = self.get_next_state()

        # print("", *self.states_set, sep="\n   ")
        from pprint import pprint
        pprint(self.states_set)

        return TDFA[E](
            tnfa.alphabet,
            tnfa.tags,
            self.states_set,
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
            tnfa_p = self.single_mapped_sym.get((tnfa_state, symbol), set())
            for p in tnfa_p:
                result[p] = Configuration(conf.registers, conf.lookahead_tags)
        return result

    def precedence(self, confs: DetConfs) -> DetPrecs:
        return list(confs.keys())

    def add_state(self, regops: RegOps) -> DetState:
        state = DetState(self.get_next_state(), self.confs, self.precs)
        if state in self.states_set:
            return state

        mapped_state = self.map_to_existing_state(state, regops)
        if mapped_state is not None:
            return mapped_state

        self.states.append(state)
        self.states_set.add(state)
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

        ignore = set()
        for i, regop in enumerate(regops):
            if regop.target in reg_to_reg1:
                regops[i].target = reg_to_reg1[regop.target]
                ignore.add(regop.target)
            # new_regops[i].target = reg_to_reg1.pop(regop.target)

        for j, i in reg_to_reg1.items():
            if j == i or j in ignore:
                continue
            regops.append(CopyOp(i, j))

        return topological_sort(regops)

    def get_final_regops(self, conf: Configuration) -> RegOps:
        result = []
        for tag in self.tnfa.tags:
            lt = conf.lookahead_tags.get(tag)
            if lt is not None:
                v = self.regop_rhs(conf.registers, lt, tag)
                i = self.final_registers[tag]
                result.append(SetOp(i, v))
        return result

    def get_transition_regops(
        self, v_map: dict[tuple[Tag, RegVal], Register]
    ) -> RegOps:
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
        else:
            indegree[regop.target] = 0

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


def tnfa_to_tdfa(tnfa: TNFA[E]) -> TDFA[E]:
    det = DeterminableTNFA()
    return det.determinization(tnfa)


@dataclass
class TDFA(Generic[E]):
    """
    Tagged Deterministic Finite Automaton
    """

    alphabet: set[E] = field(repr=False)
    tags: set[Tag]
    states: set[DetState]
    initial_state: DetState
    final_states: set[DetState]
    registers: set[Register]
    final_registers: dict[Tag, Register]
    transition_function: dict[tuple[DetState, E], tuple[DetState, RegOps]]
    final_function: dict[DetState, RegOps]
