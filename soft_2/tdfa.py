from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Sequence
from collections import deque, defaultdict
from copy import deepcopy
from pprint import pprint
from pathlib import Path
from expand_ast import NGroup2Tags
from tnfa import TNFA, OrdMapEpsTrans, MapSymTrans, Tag, AnyTag, FixedTag, Priority
from enum import Enum, auto
import tnfa


State = int
Register = int
E = TypeVar("E")


class RegVal(Enum):
    NOTHING = auto()
    CURRENT = auto()

    def __str__(self) -> str:
        if self is RegVal.NOTHING:
            return "n"
        return "p"


@dataclass(eq=True)
class Configuration:
    registers: dict[Tag, Register] = field(default_factory=dict)
    transition_tags: dict[Tag, bool] = field(default_factory=dict, compare=False)
    lookahead_tags: dict[Tag, bool] = field(default_factory=dict)

    def set_lookahead_tag(self, tag: Tag | None):
        if tag is None:
            return

        if tag > 0:
            self.lookahead_tags[tag] = True
        else:
            self.lookahead_tags[-tag] = False


def confs_as_table(confs) -> str:
    result = f"| state | tags registers | transition | lookahead |\n"
    for s, conf in confs.items():
        result += f"| {s} | {conf.registers} | {conf.transition_tags} | {conf.lookahead_tags} |\n"
    return result


DetConfs = dict[tnfa.State, Configuration]
DetPrecs = list[tnfa.State]


@dataclass(eq=True)
class DetState:
    id: State = field(compare=False)
    confs: DetConfs
    precs: DetPrecs

    def as_table(self) -> str:
        first = result = f"\n| TDFA state {self.id} |\n"
        result += confs_as_table(self.confs)
        result += "-"*len(first) + "\n"
        return result


@dataclass(unsafe_hash=True)
class SetOp:
    target: Register
    value: RegVal

    def __repr__(self) -> str:
        return f"set({self.target} <= {self.value})"


@dataclass(unsafe_hash=True)
class CopyOp:
    target: Register
    source: Register
    do_append: bool = False

    def __repr__(self) -> str:
        return f"copy({self.target} <= {self.source})"


RegOp = SetOp | CopyOp
RegOps = list[RegOp]


@dataclass
class DeterminableTNFA(Generic[E]):
    verbose: bool = False
    states: list[DetState] = field(default_factory=list)
    state_map: dict[State, DetState] = field(default_factory=dict)
    initial_state: State = field(init=False)
    final_states: set[State] = field(default_factory=set)
    transition_function: dict[tuple[State, E], tuple[State, RegOps]] = field(
        default_factory=dict
    )
    final_function: dict[State, RegOps] = field(default_factory=dict)

    tnfa: TNFA[E] = field(init=False)
    single_mapped_sym: MapSymTrans[E] = field(default_factory=MapSymTrans[E])
    ordered_eps: OrdMapEpsTrans = field(default_factory=OrdMapEpsTrans)
    confs: DetConfs = field(default_factory=DetConfs)
    precs: DetPrecs = field(default_factory=DetPrecs)
    registers: set[Register] = field(default_factory=set)
    final_registers: dict[Tag, Register] = field(default_factory=dict)
    current_reg: Register = -1
    current_state: State = -1

    def get_next_reg(self):
        self.current_reg += 1
        self.registers.add(self.current_reg)
        return self.current_reg

    def get_next_state(self):
        self.current_state += 1
        return self.current_state

    def undo_next_state(self):
        self.current_state -= 1

    def determinization(self, tnfa: TNFA[E]):
        self.tnfa = tnfa
        self.ordered_eps = tnfa.get_ordered_mapped_epsilon_transitions()
        self.single_mapped_sym = tnfa.get_mapped_symbol_transitions()

        r0 = {tag: -i for i, tag in enumerate(tnfa.tags)}
        self.final_registers = {tag: self.get_next_reg() for tag in tnfa.tags}
        self.confs = {tnfa.initial_state: Configuration(r0)}
        self.confs = self.epsilon_closure(self.confs)
        self.precs = self.precedence(self.confs)
        self.initial_state = self.add_state([]).id
        if self.verbose:
            print(self.state_map[self.initial_state].as_table())

        for state in self.states:
            v_map: dict[tuple[Tag, RegVal], Register] = {}

            if self.verbose:
                print("Generating from", state.id, "state")
            for symbol in tnfa.alphabet:
                c1 = self.confs = self.step_on_symbol(state, symbol)
                self.confs = self.epsilon_closure(self.confs)
                if not len(self.confs):
                    continue
                regops = self.get_transition_regops(v_map)
                self.precs = self.precedence(self.confs)
                next_state = self.add_state(regops)
                self.transition_function[(state.id, symbol)] = (next_state.id, regops)
                if self.verbose:
                    print(symbol, confs_as_table(c1))
                    print(next_state.as_table())
                    print()

        ordered_states = []
        for i in range(len(self.states)):
            ordered_states.append(self.state_map[i])

        return TDFA[E](
            tnfa.alphabet,
            tnfa.tags,
            ordered_states,
            self.initial_state,
            self.final_states,
            self.registers,
            self.final_registers,
            self.transition_function,
            self.final_function,
            tnfa.named_groups_to_tags,
            tnfa.miltitags,
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
                result[p] = Configuration(deepcopy(conf.registers), deepcopy(conf.lookahead_tags), dict())
        return result

    def precedence(self, confs: DetConfs) -> DetPrecs:
        return list(confs.keys())

    def add_state(self, regops: RegOps) -> DetState:
        state = DetState(self.get_next_state(), self.confs, self.precs)
        for state2 in self.states:
            if state == state2:
                self.undo_next_state()
                return state2

        mapped_state = self.map_to_existing_state(state, regops)
        if mapped_state is not None:
            if self.verbose:
                print("Map to", mapped_state.id)
            self.undo_next_state()
            return mapped_state

        self.states.append(state)
        self.state_map[state.id] = state
        for tnfa_state, conf in state.confs.items():
            if tnfa_state == self.tnfa.final_state:
                self.final_states.add(state.id)
                self.final_function[state.id] = self.get_final_regops(conf)
        return state

    def map_to_existing_state(self, state: DetState, regops: RegOps) -> DetState | None:
        for mapped_state in self.states:
            if self.verbose:
                print("mapping", state.id, "to", mapped_state.id)
            if self.map_state(state, mapped_state, regops):
                if self.verbose:
                    print("    yes map")
                return mapped_state
        return None

    def map_state(self, state: DetState, to_state: DetState, regops: RegOps) -> bool:
        if state.confs.keys() != to_state.confs.keys():
            if self.verbose:
                print("    no map", "coz keys")
            return False

        if not all(
            conf1.lookahead_tags == conf2.lookahead_tags
            for conf1, conf2 in zip(state.confs.values(), to_state.confs.values())
        ):
            if self.verbose:
                print("    no map", "coz lookahead_tags")
            return False

        if state.precs != to_state.precs:
            if self.verbose:
                print("    no map", "coz precs")
            return False

        reg_to_reg1 = dict[Register, Register]()
        reg_to_reg2 = dict[Register, Register]()

        for conf1, conf2 in zip(state.confs.values(), to_state.confs.values()):
            for tag in self.tnfa.tags:
                if conf1.lookahead_tags.get(tag) is None or tag in self.tnfa.miltitags:
                    i = conf1.registers[tag]
                    j = conf2.registers[tag]
                    m_i = reg_to_reg1.get(i, None)
                    m_j = reg_to_reg2.get(j, None)
                    if m_i is None and m_j is None:
                        reg_to_reg1[i] = j
                        reg_to_reg2[j] = i
                    elif (
                        ((m_i is not None) and (m_i != j))
                        or ((m_j is not None) and (m_j != i))
                    ):
                        if self.verbose:
                            print("    no map", f"coz not bijection {m_i=} != {j=} or {m_j=} != {i=}")
                        return False

        for i, regop in enumerate(regops):
            if regop.target in reg_to_reg1:
                regops[i].target = reg_to_reg1.pop(regop.target)

        for j, i in reg_to_reg1.items():
            if j == i:
                continue
            regops.append(CopyOp(i, j))

        return topological_sort(regops)

    def get_final_regops(self, conf: Configuration) -> RegOps:
        result = []
        for tag in self.tnfa.tags:
            i = self.final_registers[tag]
            lt = conf.lookahead_tags.get(tag)
            if lt is not None:
                v = self.regop_rhs(conf.registers, lt, tag)
                result.append(SetOp(i, v))
            else:
                j = conf.registers[tag]
                result.append(CopyOp(i, j))
        
        return result

    def get_transition_regops(
        self, v_map: dict[tuple[Tag, RegVal], Register]
    ) -> RegOps:
        result = []
        added = set()

        for conf in self.confs.values():
            for tag in self.tnfa.tags:
                ht = conf.transition_tags.get(tag)
                if ht is not None:
                    v = self.regop_rhs(conf.registers, ht, tag)
                    i = v_map.get((tag, v))
                    if i is None:
                        i = v_map[(tag, v)] = self.get_next_reg()
                    conf.registers[tag] = i
                    op = SetOp(i, v)
                    if op not in added:
                        result.append(op)
                        added.add(op)

        return result

    def regop_rhs(self, registers: dict[Tag, Register], hist: bool, tag: Tag) -> RegVal:
        # assume every tag is multi-tag
        # self.tnfa.miltitags ?
        # return (registers[tag], hist)
        # FIXME: IDK what this function does
        if hist:
            return RegVal.CURRENT
        else:
            return RegVal.NOTHING


def topological_sort(regops: RegOps) -> bool:
    set_ops = list[SetOp]()
    other_ops = list[CopyOp]()

    for regop in regops:
        if isinstance(regop, SetOp):
            set_ops.append(regop)
        else:
            other_ops.append(regop)

    # print("topsort start", regops, set_ops, other_ops)
    result = list[RegOp]()

    target_to_other_ops = defaultdict[Register, list[CopyOp]](list)
    indegree = defaultdict[Register, int](int)
    graph = defaultdict[Register, set[Register]](set)
    for regop in other_ops:
        graph[regop.source].add(regop.target)
        target_to_other_ops[regop.source].append(regop)
        indegree[regop.source]
        indegree[regop.target] += 1

    nodes = deque((target for target, count in indegree.items() if count == 0))
    visited = set[Register]()

    while nodes:
        n = nodes.pop()
        result.extend(target_to_other_ops[n])
        for m in graph.pop(n, set()):
            indegree[m] -= 1
            if indegree[m] == 0 and m not in visited:
                nodes.append(m)
                visited.add(m)

    nontrivial_cycle = any(outgoing - {target} for target, outgoing in graph.items())

    result.extend(set_ops)
    # if nontrivial_cycle:
    #     print("    no map coz nontrivial_cycle")
    # print("topsort end", result, nontrivial_cycle, graph)
    regops[:] = result
    return not nontrivial_cycle


def topological_sort_original(regops: RegOps) -> bool:
    indegree = {}

    for regop in regops:
        if isinstance(regop, CopyOp):
            indegree[regop.source] = 0
        indegree[regop.target] = 0

    for regop in regops:
        if isinstance(regop, CopyOp):
            indegree[regop.source] += 1

    result = []

    nontrivial_cycle = False
    # print("topsort start", regops)
    queue = deque(regops)

    while queue:
        something_were_added = False

        for _ in range(len(queue)):
            regop = queue.pop()
            if indegree[regop.target] == 0:
                result.append(regop)
                something_were_added = True
                if isinstance(regop, CopyOp):
                    indegree[regop.source] -= 1
            else:
                queue.append(regop)

        if not something_were_added and queue:
            if any(
                regop.target != regop.source
                for regop in queue
                if isinstance(regop, CopyOp)
            ):
                nontrivial_cycle = True
            result.extend(queue)
            break  # only cycles left

    # print("topsort end", result)
    regops[:] = result
    # if nontrivial_cycle:
    #     print("    no map coz nontrivial_cycle")
    return not nontrivial_cycle


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
    states: list[DetState]
    initial_state: State
    final_states: set[State]
    registers: set[Register]
    final_registers: dict[Tag, Register]
    transition_function: dict[tuple[State, E], tuple[State, RegOps]]
    final_function: dict[State, RegOps]

    named_groups_to_tags: NGroup2Tags
    miltitags: set[Tag]

    def dumps_dot(self) -> str:
        result = []
        result.append("digraph G {\n")
        result.append('node [label="", shape=circle, style=filled];\n\n')

        for state in self.states:
            if state.id == self.initial_state:
                result.append(f'n{state.id} [label="{state.id}", shape=doublecircle];\n')
            elif state.id in self.final_states:
                result.append(f'n{state.id} [label="{state.id}", shape=doublecircle];\n')
            else:
                result.append(f'n{state.id} [label="{state.id}"];\n')

            if state.id in self.final_function:
                result.append(f'n{state.id}_fin [style = invis];\n')

        for (q, s), (p, o) in self.transition_function.items():
            ops = "".join(f"\\n{op}" for op in o)
            result.append(f'n{q} -> n{p} [label="{s}/{ops}"];\n')

        for q, o in self.final_function.items():
            ops = "".join(f"\\n{op}" for op in o)
            result.append(
                f'n{q} -> n{q}_fin [label="{ops}"];\n'
            )

        result.append("}\n")

        return "".join(result)

    def dump_dot(self, path: Path | str) -> Path:
        path = Path(path)
        path.write_text(self.dumps_dot(), encoding="utf-8")
        return path

    def as_simulatable(self) -> SimulatableTDFA[E]:
        # regs = [MultipleRegisterStorage() if tag in self.miltitags else SingleRegisterStorage() for tag in self.final_registers]
        # regs += [SingleRegisterStorage() for _ in range(len(regs), len(self.registers))]
        regs: list[RegisterStorage]
        if self.miltitags:
            regs = [MultipleRegisterStorage() for _ in self.registers]
        else:
            regs = [SingleRegisterStorage() for _ in self.registers]
        return SimulatableTDFA[E](
            self.initial_state,
            self.final_states,
            self.final_registers,
            self.transition_function,
            self.final_function,
            regs,
            self.named_groups_to_tags,
        )


@dataclass
class SingleRegisterStorage:
    value: int | None = None

    def set(self, value: int | None) -> None:
        self.value = value

    def get_last(self) -> int | None:
        return self.value

    def get_all(self) -> Sequence[int | None]:
        return [self.value]

    def clear(self) -> None:
        self.value = None


@dataclass
class MultipleRegisterStorage:
    values: deque[int | None] = field(default_factory=deque)

    def set(self, value: int | None) -> None:
        self.values.append(value)

    def get_last(self) -> int | None:
        return self.values[-1]

    def get_all(self) -> Sequence[int | None]:
        return self.values

    def clear(self) -> None:
        self.values.clear()


RegisterStorage = SingleRegisterStorage | MultipleRegisterStorage


@dataclass
class SimulatableTDFA(Generic[E]):
    initial_state: State
    final_states: set[State]
    final_registers: dict[Tag, Register]
    transition_function: dict[tuple[State, E], tuple[State, RegOps]]
    final_function: dict[State, RegOps]

    registers: list[RegisterStorage]
    named_groups_to_tags: NGroup2Tags

    def execute_regops(self, index: int, regops: RegOps) -> None:
        for regop in regops:
            if isinstance(regop, SetOp):
                if regop.value is RegVal.NOTHING:
                    self.registers[regop.target].set(None)
                else:
                    self.registers[regop.target].set(index)
            elif isinstance(regop, CopyOp):
                self.registers[regop.target].set(self.registers[regop.source].get_last())

    def get_register_storage(self, tag: AnyTag) -> tuple[RegisterStorage, int]:
        if isinstance(tag, FixedTag):
            return self.registers[self.final_registers[tag.origin]], tag.shift
        else:
            return self.registers[self.final_registers[tag]], 0

    def gather_matches(self, word: Sequence[E]) -> dict[str, list[str]]:
        matches = defaultdict(list)
        for name, (start, end) in self.named_groups_to_tags.items():
            start_store, start_offset = self.get_register_storage(start)
            end_store, end_offset = self.get_register_storage(end)
            for start_ind, end_ind in zip(start_store.get_all(), end_store.get_all()):
                if start_ind is not None and end_ind is not None:
                    matches[name].append(word[start_ind + start_offset: end_ind + end_offset])
                else:
                    matches[name].append(None)
        return dict(matches)

    def simulate(self, word: Sequence[E]) -> dict[str, list[str]] | None:
        state = self.initial_state
        for reg in self.registers:
            reg.clear()

        for i, symbol in enumerate(word):
            res = self.transition_function.get((state, symbol))
            if res is None:
                return None

            state, regops = res
            self.execute_regops(i, regops)

        if state not in self.final_states:
            return None

        regops = self.final_function.get(state)
        if regops is not None:
            self.execute_regops(len(word), regops)

        return self.gather_matches(word)
