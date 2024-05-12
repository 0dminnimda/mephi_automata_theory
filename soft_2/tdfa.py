from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Sequence, Iterable
from collections import deque, defaultdict
from copy import deepcopy
from pprint import pprint
from parser import ANY_SYMBOL_RE
from pathlib import Path
from simplify_ast import NGroup2Tags
import classes as ast
from tnfa import (
    TNFA,
    OrdMapEpsTrans,
    DblMapSymTrans,
    Tag,
    AnyTag,
    FixedTag,
    Matcher,
    Priority,
    dump_matcher,
)
from enum import Enum, auto
from helpers import split_overlapping_intervals
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

    def as_suffix(self) -> str:
        if self is RegVal.NOTHING:
            return "↓"  # "V"
        return "↑"  # "A"

    def evaluate(self, index: int) -> int | None:
        if self is RegVal.NOTHING:
            return None
        return index


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
        result += "-" * len(first) + "\n"
        return result


@dataclass(unsafe_hash=True)
class SetOp:
    target: Register
    value: RegVal

    def __repr__(self) -> str:
        return f"{self.target}{self.value.as_suffix()}"


History = RegVal  # actually a list[RegVal], but it's probably fine for now
RegOpRightHandSide = tuple[Register | RegVal, History | None]


@dataclass(unsafe_hash=True)
class CopyOp:
    target: Register
    source: Register
    history: History | None = None

    def __repr__(self) -> str:
        suff = "" if self.history is None else self.history.as_suffix()
        return f"{self.target}={self.source}{suff}"


RegOp = SetOp | CopyOp
RegOps = list[RegOp]
RegOpsTuple = tuple[RegOp, ...]


@dataclass
class DeterminableTNFA(Generic[E]):
    """
    Basically Laurikari algorithm
    """

    verbose: bool = False
    states: list[DetState] = field(default_factory=list)
    state_map: dict[State, DetState] = field(default_factory=dict)
    initial_state: State = field(init=False)
    final_states: set[State] = field(default_factory=set)
    transition_function: dict[tuple[State, Matcher], tuple[State, RegOps]] = field(
        default_factory=dict
    )
    final_function: dict[State, RegOps] = field(default_factory=dict)

    tnfa: TNFA[E] = field(init=False)
    double_mapped_sym: DblMapSymTrans = field(default_factory=DblMapSymTrans)
    ordered_eps: OrdMapEpsTrans = field(default_factory=OrdMapEpsTrans)
    confs: DetConfs = field(default_factory=DetConfs)
    precs: DetPrecs = field(default_factory=DetPrecs)
    registers: set[Register] = field(default_factory=set)
    final_registers: dict[Tag, Register] = field(default_factory=dict)
    tag_to_regs: defaultdict[Tag, list[Register]] = field(
        default_factory=lambda: defaultdict(list)
    )
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

        self.update_tnfa_transitions_as_accepting()
        all_matchers = self.get_all_unique_matchers(m for _, m, _ in self.tnfa.symbol_transitions)

        self.ordered_eps = tnfa.get_ordered_mapped_epsilon_transitions()
        self.double_mapped_sym = tnfa.get_double_mapped_symbol_transitions()

        r0 = {tag: self.get_next_reg() for tag in tnfa.tags}
        self.final_registers = {tag: self.get_next_reg() for tag in tnfa.tags}
        for tag, reg in self.final_registers.items():
            self.tag_to_regs[tag].append(reg)
        self.confs = {tnfa.initial_state: Configuration(r0)}
        self.confs = self.epsilon_closure(self.confs)
        self.precs = self.precedence(self.confs)
        self.initial_state = self.add_state([]).id
        if self.verbose:
            print(self.state_map[self.initial_state].as_table())

        for state in self.states:
            if self.verbose:
                print("Generating from", state.id, "state")

            v_map: dict[tuple[Tag, RegOpRightHandSide], Register] = {}
            for matcher in all_matchers:
                c1 = self.confs = self.step_on_symbol(state, matcher)
                self.confs = self.epsilon_closure(self.confs)
                if not len(self.confs):
                    continue
                regops = self.get_transition_regops(v_map)
                self.precs = self.precedence(self.confs)
                next_state = self.add_state(regops)
                self.transition_function[(state.id, matcher)] = (next_state.id, regops)
                if self.verbose:
                    print(dump_matcher(matcher), confs_as_table(c1))
                    print(next_state.as_table())
                    print(regops)
                    print()

        if self.verbose:
            print("FINAL")
            for state in self.states:
                print(state.as_table())
                print()

        ordered_states = []
        for i in range(len(self.states)):
            ordered_states.append(self.state_map[i])

        if self.verbose:
            print(self.tag_to_regs)

        self.merge_transitions()

        return TDFA[E](
            tnfa.tags,
            ordered_states,
            self.initial_state,
            self.final_states,
            self.registers,
            self.final_registers,
            self.transition_function,
            self.final_function,
            dict(self.tag_to_regs),
            tnfa.named_groups_to_tags,
            tnfa.multitags,
        )

    @staticmethod
    def separate_symbol_and_other_matchers(
        mathcers: list[Matcher],
    ) -> tuple[list[ast.SymbolRanges], list[Matcher]]:
        range_matchers = list[ast.SymbolRanges]()
        other_matchers = list[Matcher]()
        for matcher in mathcers:
            if isinstance(matcher, ast.SymbolRanges):
                range_matchers.append(matcher)
            else:
                other_matchers.append(matcher)
        return range_matchers, other_matchers

    def merge_transitions(self):
        all_trans: defaultdict[tuple[State, State], dict[RegOpsTuple, list[Matcher]]]
        all_trans = defaultdict(lambda: defaultdict(list))

        for (incoming, matcher), (
            outcoming,
            regops,
        ) in self.transition_function.items():
            all_trans[(incoming, outcoming)][tuple(regops)].append(matcher)

        new_transition_function = dict()
        for (incoming, outcoming), mapping in all_trans.items():
            for regops, matchers in mapping.items():
                symbol_matchers, matchers = self.separate_symbol_and_other_matchers(
                    matchers
                )
                symbol_matchers = ast.SymbolRanges.merge(symbol_matchers)
                if symbol_matchers:
                    assert (
                        len(symbol_matchers) == 1
                    ), "Only positive matchers should be used for determinization"
                    matchers.append(symbol_matchers[0])

                for matcher in matchers:
                    new_transition_function[(incoming, matcher)] = (outcoming, regops)

        self.transition_function = new_transition_function

    def update_tnfa_transitions_as_accepting(self):
        symbol_transitions = set()
        for a, matcher, b in self.tnfa.symbol_transitions:
            if isinstance(matcher, ast.SymbolRanges):
                matcher = matcher.minimized_as_accepting()
            symbol_transitions.add((a, matcher, b))
        self.tnfa.symbol_transitions = symbol_transitions

    @staticmethod
    def get_all_unique_matchers(matchers: Iterable[Matcher]) -> list[Matcher]:
        # 1. All non-symbol-ranges (group matchers) are unique
        # 2. In the symbol-ranges we have to extract non-overlapping ones,
        #    for exmaple, if we have [a-d], [a-ho], [e-h] we need to check have [a-d], [e-h], [o]

        range_matchers = list[tuple[str, str]]()
        other_matchers = list[Matcher]()
        for matcher in matchers:
            if isinstance(matcher, ast.SymbolRanges):
                range_matchers.extend(matcher.ranges)
            else:
                other_matchers.append(matcher)

        non_overlaping = split_overlapping_intervals(
            list(ast.SymbolRanges.ranges_as_intervals(range_matchers))
        )
        range_matchers = [
            ast.SymbolRanges((it,), True)
            for it in ast.SymbolRanges.intervals_as_ranges(non_overlaping)
        ]

        return range_matchers + other_matchers

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

    @staticmethod
    def matcher_can_pass(matcher_base: Matcher, matcher_passing: Matcher) -> bool:
        if matcher_base is matcher_passing:
            return True

        if isinstance(matcher_base, ast.SymbolRanges) and isinstance(
            matcher_passing, ast.SymbolRanges
        ):
            return matcher_base.covers(matcher_passing)

        return False

    def step_on_symbol(self, state: DetState, matcher: Matcher) -> DetConfs:
        result = DetConfs()
        for tnfa_state in state.precs:
            conf = state.confs[tnfa_state]
            for matcher_base, next_tnfa_state in self.double_mapped_sym.get(
                tnfa_state, dict()
            ).items():
                if self.matcher_can_pass(matcher_base, matcher):
                    result[next_tnfa_state] = Configuration(
                        deepcopy(conf.registers), deepcopy(conf.lookahead_tags), dict()
                    )
        return result

    def precedence(self, confs: DetConfs) -> DetPrecs:
        # We create the nodes in such a way that the nodes
        # that are should be considered later in leftmost greed
        # are numerically smaller
        return sorted(confs.keys())

    def add_state(self, regops: RegOps) -> DetState:
        state = DetState(self.get_next_state(), self.confs, self.precs)
        for state2 in self.states:
            if state == state2:
                if self.verbose:
                    print("Mapping to equal!")
                self.undo_next_state()
                return state2

        mapped_state = self.map_to_existing_state(state, regops)
        if mapped_state is not None:
            if self.verbose:
                print("Map to", mapped_state.id, "using", regops)
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

        if self.verbose:
            print("Mapping", regops, reg_to_reg1)
        for conf1, conf2 in zip(state.confs.values(), to_state.confs.values()):
            for tag in self.tnfa.tags:
                if conf1.lookahead_tags.get(tag) is None or tag in self.tnfa.multitags:
                    i = conf1.registers[tag]
                    j = conf2.registers[tag]
                    m_i = reg_to_reg1.get(i, None)
                    m_j = reg_to_reg2.get(j, None)
                    if m_i is None and m_j is None:
                        reg_to_reg1[i] = j
                        reg_to_reg2[j] = i
                    elif ((m_i is not None) and (m_i != j)) or (
                        (m_j is not None) and (m_j != i)
                    ):
                        if self.verbose:
                            print(
                                "    no map",
                                f"coz not bijection {m_i=} != {j=} or {m_j=} != {i=}",
                            )
                        return False

        if self.verbose:
            print("Mapping", regops, reg_to_reg1)
        for i, regop in enumerate(regops):
            if regop.target in reg_to_reg1:
                regops[i].target = reg_to_reg1.pop(regop.target)

        if self.verbose:
            print("Mapping", regops, reg_to_reg1)
        for j, i in reg_to_reg1.items():
            if j == i:
                continue
            regops.append(CopyOp(i, j))

        if self.verbose:
            print("Mapping to sort", regops, reg_to_reg1)
        return topological_sort(regops)

    def get_final_regops(self, conf: Configuration) -> RegOps:
        result = []
        for tag in self.tnfa.tags:
            i = self.final_registers[tag]
            lt = self.history(conf.lookahead_tags, tag)
            if lt is not None:
                v = self.regop_rhs(conf.registers, lt, tag)
                result.append(self.make_regop_from_rhs(i, v))
            else:
                j = conf.registers[tag]
                result.append(CopyOp(i, j))

        return result

    def get_transition_regops(
        self, v_map: dict[tuple[Tag, RegOpRightHandSide], Register]
    ) -> RegOps:
        result = []
        added = set()

        if self.verbose:
            print(self.tnfa.tags)
            print(confs_as_table(self.confs))
        for conf in self.confs.values():
            for tag in self.tnfa.tags:
                ht = self.history(conf.transition_tags, tag)
                if ht is not None:
                    v = self.regop_rhs(conf.registers, ht, tag)
                    i = v_map.get((tag, v))
                    if i is None:
                        i = v_map[(tag, v)] = self.get_next_reg()
                        self.tag_to_regs[tag].append(i)
                    conf.registers[tag] = i
                    op = self.make_regop_from_rhs(i, v)
                    if op not in added:
                        result.append(op)
                        added.add(op)

        if self.verbose:
            print("get_transition_regops ->", result)
        return result

    @staticmethod
    def history(conf_tags: dict[Tag, bool], tag: Tag) -> History | None:
        val = conf_tags.get(tag)
        if val is None:
            return None

        if val:
            return RegVal.CURRENT
        else:
            return RegVal.NOTHING

    def regop_rhs(
        self, registers: dict[Tag, Register], history: History, tag: Tag
    ) -> RegOpRightHandSide:
        if tag in self.tnfa.multitags:
            return registers[tag], history
        if history:
            return RegVal.CURRENT, None
        else:
            return RegVal.NOTHING, None

    @staticmethod
    def make_regop_from_rhs(target: Register, rhs: RegOpRightHandSide) -> RegOp:
        val, hist = rhs
        if isinstance(val, RegVal):
            return SetOp(target, val)
        return CopyOp(target, val, hist)


def topological_sort_not_fully_correct(regops: RegOps) -> bool:
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


def topological_sort(regops: RegOps) -> bool:
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
    tdfa = det.determinization(tnfa)
    tdfa.minimize_states()
    return tdfa


SimTranFunc = dict[State, list[tuple[Matcher, State, RegOps]]]
TDFATranFunc = dict[tuple[State, Matcher], tuple[State, RegOps]]


@dataclass
class TDFA(Generic[E]):
    """
    Tagged Deterministic Finite Automaton
    """

    # alphabet: set[E] = field(repr=False)  # we don't need that either
    tags: set[Tag]
    states: list[DetState]
    initial_state: State
    final_states: set[State]
    registers: set[Register]
    final_registers: dict[Tag, Register]
    transition_function: TDFATranFunc
    final_function: dict[State, RegOps]

    tag_to_regs: dict[Tag, list[Register]]
    named_groups_to_tags: NGroup2Tags
    multitags: set[Tag]

    def dumps_dot(self) -> str:
        result = []
        result.append("digraph G {\n")
        result.append("rankdir=LR\n")
        result.append(
            'node [label="", shape=circle, style=filled, fontname=Courier];\n'
        )
        result.append("edge[arrowhead=vee fontname=Courier]\n")
        result.append("\n")
        result.append(
            f'n [shape=point xlabel="Start"] n -> n{self.initial_state} [style=dotted]\n'
        )

        for (q, s), (p, o) in self.transition_function.items():
            ops = " ".join(f"{op}" for op in o)
            if ops:
                ops = "\\n" + ops
            result.append(f'n{q} -> n{p} [label="{dump_matcher(s)}{ops}"];\n')

        for q, o in self.final_function.items():
            ops = " ".join(str(op) for op in o)
            result.append(
                f'subgraph {{ rank=same n{q} dr{q} [shape=rect style=dotted fillcolor=transparent label="{ops}"] n{q}:s -> dr{q}:n [style=dotted minlen=0]}}\n'
            )

        for state in self.states:
            preoperties = []
            if state.id == self.initial_state:
                preoperties.append("style=empty")
            if state.id in self.final_states:
                preoperties.append("shape=doublecircle")
            preoperties = " ".join(preoperties)
            result.append(f'n{state.id} [label="{state.id}" {preoperties}];\n')

            if state.id in self.final_function:
                result.append(f"n{state.id}_fin [style = invis];\n")

        result.append("}\n")

        return "".join(result)

    def dump_dot(self, path: Path | str) -> Path:
        path = Path(path)
        path.write_text(self.dumps_dot(), encoding="utf-8")
        return path

    @staticmethod
    def remap_regops(regops: RegOps, mapping: dict[Register, Register]) -> RegOps:
        result = []
        setops = {}
        for regop in regops:
            if isinstance(regop, SetOp):
                regop = SetOp(mapping[regop.target], regop.value)
                ind = setops.get(regop.target)
                if ind is not None:
                    if regop.value is RegVal.CURRENT:
                        result[ind] = regop
                    continue
                setops[regop.target] = len(result)
            else:
                regop = CopyOp(mapping[regop.target], mapping[regop.source])
                if regop.target == regop.source:
                    continue
            result.append(regop)
        return result

    def collapse_extra_registers(self):
        self.registers = set(self.final_registers.values())
        old_reg_2_new = {}
        for tag, regs in self.tag_to_regs.items():
            for reg in regs:
                old_reg_2_new[reg] = self.final_registers[tag]
        self.tag_to_regs = {tag: [reg] for tag, reg in self.final_registers.items()}

        transition_function = {}
        for tup, (state, regops) in self.transition_function.items():
            transition_function[tup] = (state, self.remap_regops(regops, old_reg_2_new))
        self.transition_function = transition_function

        final_function = {}
        for state, regops in self.final_function.items():
            final_function[state] = self.remap_regops(regops, old_reg_2_new)
        self.final_function = final_function

    def to_simulation_transition_function(self) -> SimTranFunc:
        result: SimTranFunc = {}
        for (q, s), (p, o) in self.transition_function.items():
            result[q] = result.get(q, [])
            result[q].append((s, p, o))

        sorted_result: SimTranFunc = {}
        for q, things in result.items():
            symb = []
            non_symb = []
            for matcher, state, regops in things:
                if isinstance(matcher, ast.SymbolRanges):
                    symb.append((matcher, state, regops))
                else:
                    non_symb.append((matcher, state, regops))
            sorted_result[q] = non_symb + symb

        return sorted_result

    def as_simulatable(self) -> SimulatableTDFA[E]:
        # regs = [MultipleRegisterStorage() if tag in self.multitags else SingleRegisterStorage() for tag in self.final_registers]
        # regs += [SingleRegisterStorage() for _ in range(len(regs), len(self.registers))]
        regs: list[RegisterStorage]
        if self.multitags:
            regs = [MultipleRegisterStorage() for _ in self.registers]
        else:
            regs = [SingleRegisterStorage() for _ in self.registers]

        return SimulatableTDFA[E](
            self.initial_state,
            self.final_states,
            self.final_registers,
            self.to_simulation_transition_function(),
            self.final_function,
            self.tag_to_regs,
            regs,
            self.named_groups_to_tags,
        )

    def minimize_states(self):
        # using Hopcroft's algorithm, afaik

        all_trans = [(mat, tuple(regops)) for (_, mat), (_, regops) in self.transition_function.items()]
        states_set = {it.id for it in self.states}

        incoming_trans = defaultdict(set)
        for (src, mat), (dst, regops) in self.transition_function.items():
            incoming_trans[(mat, dst, tuple(regops))].add(src)

        partitions = [self.final_states, states_set - self.final_states]
        partitions = [it for it in partitions if it]
        incomplete = partitions[:]

        while incomplete:
            A = incomplete.pop()
            for matcher, regops in all_trans:
                X = set.union(*(incoming_trans[(matcher, in_state, regops)] for in_state in A))
                if len(X) == 0:
                    continue
                for i, Y in enumerate(partitions):
                    XandY = X & Y
                    YnotX = Y - X
                    if len(XandY) == 0 or len(YnotX) == 0:
                        continue
                    partitions[i : i+1] = [XandY, YnotX]

                    try:
                        index = incomplete.index(Y)
                    except ValueError:
                        index = -1

                    if index != -1:
                        incomplete[index : index+1] = [XandY, YnotX]
                    else:
                        if len(XandY) <= len(YnotX):
                            incomplete.append(XandY)
                        else:
                            incomplete.append(YnotX)

        if all(len(p) == 1 for p in partitions):
            return self

        initial_state = 0
        final_states = set()
        state_map = dict[State, State]()
        partitions.sort(key=lambda x: min(x))
        for new_state, partition in enumerate(partitions):
            if self.initial_state in partition:
                initial_state = new_state

            if self.final_states & partition:
                assert len(self.final_states & partition) == len(partition), "non-final states cannot be grouped with final ones"
                final_states.add(new_state)

            for state in partition:
                state_map[state] = new_state

        self.states = [self.states[it] for it in set(state_map.values())]
        self.initial_state = initial_state
        self.final_states = final_states
        self.transition_function = {
            (state_map[src], mat): (state_map[dst], regops)
            for (src, mat), (dst, regops) in self.transition_function.items()
        }
        self.final_function = {state_map[q]: regops for q, regops in self.final_function.items()}

        return self

    def complement(self) -> TDFA | None:
        return deepcopy(self).turn_into_complement()

    def add_temporary_error_state(self) -> tuple[State, set[State], TDFATranFunc]:
        states_set = set(it.id for it in self.states)
        error_state: State = len(self.states)
        assert error_state not in states_set, "state ids do not correspond to indices in the state list"
        states_set.add(error_state)

        new_transition_function = dict(self.transition_function)
        ranges = defaultdict[State, list[ast.SymbolRanges]](list)
        for (src, matcher), _ in self.transition_function.items():
            if not isinstance(matcher, ast.SymbolRanges):
                continue
            ranges[src].append(matcher)

        for src, matchers in ranges.items():
            # relies on the fact that the trsntisions were merged
            all_ranges = []
            for matcher in matchers:
                assert matcher.accept, "the simulatable tdfa should be merged"
                all_ranges.extend(matcher.ranges)
            matcher = ast.SymbolRanges(tuple(all_ranges), True).with_minimized_ranges()
            matcher = ast.SymbolRanges(matcher.ranges, False).minimized_as_accepting()
            new_transition_function[(src, matcher)] = (error_state, [])

        for src in states_set - ranges.keys():
            new_transition_function[(src, ANY_SYMBOL_RE)] = (error_state, [])

        new_transition_function[(error_state, ANY_SYMBOL_RE)] = (error_state, [])

        return error_state, states_set, new_transition_function

    def turn_into_complement(self) -> TDFA | None:
        # To implement a complement you need to create states
        # so all transitions over the whole alphabet will lead to some state
        # (wich can never result in any match!)
        # Now swap final states and all other states between eachother
        # now remove the states that can never lead to match and their transitions
        # voila, you have a complemented regex!

        # Shortcut: while matching if we know the previous tdfa could not have reached this state
        # then we don't care about the structure from this point on
        # We can just create one "error" state and connect all states to it, and loop inside it through all symbols

        for (_, matcher), _ in self.transition_function.items():
            if not isinstance(matcher, ast.SymbolRanges):
                return None

        error_state, states_set, new_transition_function = self.add_temporary_error_state()
        self.states.append(DetState(error_state, {}, []))
        self.transition_function = new_transition_function

        # For now I don't not remove the possibly useless states that can't lead to match

        self.final_states = states_set - self.final_states
        self.final_function = {}

        return self


@dataclass
class SingleRegisterStorage:
    value: int | None = None

    def set(self, value: int | None) -> None:
        self.value = value

    def get_last(self) -> int | None:
        return self.value

    def get_all(self) -> deque[int | None]:
        return deque([self.value])

    def copy_from(self, other: RegisterStorage) -> None:
        if isinstance(other, SingleRegisterStorage):
            self.values = other.value
            return

        values = other.values
        if values:
            self.value = values[-1]
        else:
            self.value = None

    def clear(self) -> None:
        self.value = None

    def __repr__(self) -> str:
        return f"reg({self.value})"


@dataclass
class MultipleRegisterStorage:
    values: deque[int | None] = field(default_factory=deque)

    def set(self, value: int | None) -> None:
        self.values.append(value)

    def get_last(self) -> int | None:
        if self.values:
            return self.values[-1]
        return None

    def get_all(self) -> deque[int | None]:
        return self.values

    def copy_from(self, other: RegisterStorage) -> None:
        if isinstance(other, MultipleRegisterStorage):
            self.values = other.values.copy()
        else:
            self.values = deque([other.value])

    def clear(self) -> None:
        self.values = deque()

    def __repr__(self) -> str:
        return f"reg({str(self.values)[6:-1]})"


RegisterStorage = SingleRegisterStorage | MultipleRegisterStorage


@dataclass
class SimulatableTDFA(Generic[E]):
    initial_state: State
    final_states: set[State]
    final_registers: dict[Tag, Register]
    transition_function: SimTranFunc
    final_function: dict[State, RegOps]

    tag_to_regs: dict[Tag, list[Register]]
    registers: list[RegisterStorage]
    named_groups_to_tags: NGroup2Tags

    def restore_regular_expression_via_k_path(self) -> str | None:
        transition_map: dict[tuple[State, State], list[Matcher]] = defaultdict(list)
        for inc, other in self.transition_function.items():
            for matcher, out, _ in other:
                transition_map[(inc, out)].append(matcher)

        dumpped_tran_map: dict[tuple[State, State], str] = dict()
        for key, matchers in transition_map.items():
            if any(not isinstance(m, ast.SymbolRanges) for m in matchers):
                return None  # we don't support group references yet
            matchers = {
                dump_matcher(m, escape_meta=True, escape_dot=False) for m in matchers
            }
            result = "|".join(it for it in matchers if it)
            if len(matchers) > 1:
                result = f"({result})"
            dumpped_tran_map[key] = result

        max_state: State = max(self.transition_function.keys())
        if max_state >= 500:
            return None
        memo = {}
        arguments = []
        paths = []
        for fin in self.final_states:
            last = ""
            for src, dst, K in self._k_path_generate_arguments(
                self.initial_state, fin, max_state, arguments
            )[::-1]:
                last = self._k_path_memo(src, dst, K, memo, dumpped_tran_map)
            paths.append(last)
        paths = [f"({p})" for p in paths if p is not None]
        return "|".join(set(paths))

    def _k_path_generate_arguments(
        self, source: State, target: State, K: State, prev_result
    ) -> list[tuple[State, State, State]]:
        triplet = (source, target, K)
        if triplet in prev_result:
            return prev_result
        result = [triplet]
        visited = set(prev_result + result)
        queue = deque(result)
        while queue:
            source, target, K = queue.popleft()
            if K == -1:
                triplet = (source, target, K)
                if triplet not in visited:
                    result.append(triplet)
                    visited.add(triplet)
                continue
            for triplet in [
                (source, target, K - 1),
                (source, K, K - 1),
                (K, K, K - 1),
                (K, target, K - 1),
            ]:
                if triplet not in visited:
                    queue.append(triplet)
                    result.append(triplet)
                    visited.add(triplet)
        prev_result.extend(result)
        return result

    def _k_path_memo(
        self,
        source: State,
        target: State,
        K: State,
        memo: dict[tuple[State, State, State], str | None],
        transition_map: dict[tuple[State, State], str],
    ) -> str | None:

        cache_key = (source, target, K)
        # print("in", cache_key)

        cache = memo.get(cache_key)
        if cache is not None:
            # print("cache", cache_key, cache)
            return cache

        if K == -1:
            result = transition_map.get((source, target), "")
            if not result and source != target:
                result = None
            # print("k==-1", cache_key, result)
            memo[cache_key] = result
            return result

        first_part = memo[(source, target, K - 1)]

        if source != K:
            second_first = memo[(source, K, K - 1)]
        else:
            second_first = ""

        repeating = memo[(K, K, K - 1)]
        repeating = f"({repeating})..." if repeating else ""  # ok for it to be None

        if K != target:
            second_third = memo[(K, target, K - 1)]
        else:
            second_third = ""

        if second_first is None or second_third is None:
            second_part = None
        else:
            second_part = second_first + repeating + second_third

        if first_part is not None and second_part is not None:
            if first_part == second_part:
                result = first_part
            else:
                result = f"({first_part}|{second_part})"
        elif first_part is not None:
            result = first_part
        else:
            result = second_part

        # print("gen", cache_key, result)
        memo[cache_key] = result
        return result

    def to_partial_tdfa(self) -> TDFA:
        return TDFA(
            set(self.tag_to_regs.keys()),
            list(
                DetState(it, {}, [])
                for it in set(self.transition_function.keys()) | set(self.final_states)
            ),
            self.initial_state,
            self.final_states,
            {i for i in range(len(self.registers))},
            self.final_registers,
            {
                (src, mat): (dst, op)
                for src, rest in self.transition_function.items()
                for mat, dst, op in rest
            },
            self.final_function,
            self.tag_to_regs,
            self.named_groups_to_tags,
            set(),
        )

    def execute_regops(self, index: int, regops: RegOps) -> None:
        for regop in regops:
            if isinstance(regop, SetOp):
                self.registers[regop.target].set(regop.value.evaluate(index))
            elif isinstance(regop, CopyOp):
                store = self.registers[regop.target]
                store.copy_from(self.registers[regop.source])
                if regop.history is not None:
                    store.set(regop.history.evaluate(index))

    def get_register_storage_from_tag_final(
        self, tag: AnyTag
    ) -> tuple[RegisterStorage, int]:
        if isinstance(tag, FixedTag):
            return self.registers[self.final_registers[tag.origin]], tag.offset
        else:
            return self.registers[self.final_registers[tag]], 0

    def get_register_storage_from_tag_all(
        self, tag: AnyTag
    ) -> tuple[Iterable[RegisterStorage], int]:
        if isinstance(tag, FixedTag):
            return (
                self.registers[it] for it in self.tag_to_regs[tag.origin]
            ), tag.offset
        else:
            return (self.registers[it] for it in self.tag_to_regs[tag]), 0

    def get_one_register_value_from_tag_all(self, tag: AnyTag) -> int | None:
        start_stores, start_offset = self.get_register_storage_from_tag_all(tag)
        start_ind = None
        for start_store in start_stores:
            for start_ind in reversed(start_store.get_all()):
                if start_ind is not None:
                    break

            if start_ind is not None:
                return start_ind + start_offset
        return None

    def gather_matches(self, word: str) -> dict[str, list[str]]:
        matches = defaultdict(list)
        for name, (start, end) in self.named_groups_to_tags.items():
            start_store, start_offset = self.get_register_storage_from_tag_final(start)
            end_store, end_offset = self.get_register_storage_from_tag_final(end)
            for start_ind, end_ind in zip(start_store.get_all(), end_store.get_all()):
                if start_ind is not None and end_ind is not None:
                    matches[name].append(
                        word[start_ind + start_offset : end_ind + end_offset]
                    )
                else:
                    matches[name].append(None)
        return dict(matches)

    def run_matcher(self, matcher: Matcher, word: str, index: int) -> int | None:
        if isinstance(matcher, ast.SymbolRanges):
            if matcher.matches(word[index]):
                return index + 1
            return None
        else:
            start_ind = self.get_one_register_value_from_tag_all(matcher.start_tag)
            end_ind = self.get_one_register_value_from_tag_all(matcher.end_tag)

            if start_ind is None or end_ind is None:
                return None

            group_match = word[start_ind:end_ind]
            if word.startswith(group_match, index):
                return index + len(group_match)
            else:
                return None

    def find_next_transition(
        self, state: State, word: str, index: int
    ) -> tuple[int, State, RegOps] | None:
        for matcher, next_state, regops in self.transition_function.get(state, []):
            next_index = self.run_matcher(matcher, word, index)
            if next_index is not None:
                return next_index, next_state, regops
        return None

    def match_whole_string(
        self, word: str, initial_index: int = 0
    ) -> dict[str, list[str]] | None:
        state = self.initial_state
        for reg in self.registers:
            reg.clear()

        index = initial_index
        while index < len(word):
            res = self.find_next_transition(state, word, index)
            if res is None:
                return None

            next_index, state, regops = res
            self.execute_regops(index, regops)
            index = next_index

        if state not in self.final_states:
            return None

        regops = self.final_function.get(state)
        if regops is not None:
            self.execute_regops(index, regops)

        return self.gather_matches(word)

    def match_maximum_length(
        self, word: str, initial_index: int = 0, capture: bool = False
    ) -> tuple[int, dict[str, list[str]] | None]:
        state = self.initial_state
        if capture:
            for reg in self.registers:
                reg.clear()

        last_matching: tuple[State, int, list[RegisterStorage]] | None = None

        index = initial_index
        while 1:
            if state in self.final_states:
                if capture:
                    last_matching = state, index, deepcopy(self.registers)
                else:
                    last_matching = state, index, self.registers

            if index >= len(word):
                break

            res = self.find_next_transition(state, word, index)
            if res is None:
                break

            next_index, state, regops = res
            if capture:
                self.execute_regops(index, regops)
            index = next_index

        if last_matching is None:
            return initial_index, None

        state, index, self.registers = last_matching

        if capture:
            regops = self.final_function.get(state)
            if regops is not None:
                self.execute_regops(index, regops)

        if capture:
            return index, self.gather_matches(word)
        else:
            return index, {}
