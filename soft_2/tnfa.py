from __future__ import annotations

import string
import classes as ast
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Sequence, NamedTuple
from collections import deque, defaultdict
from pathlib import Path
import os
from parser import iter_unique

from copy import deepcopy


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


class SymbolTransition(NamedTuple, Generic[E]):
    source: State
    symbol: E
    target: State


class EpsilonTransition(NamedTuple):
    source: State
    priority: Priority
    tag: Tag | None
    target: State


@dataclass(frozen=True)
class NamedGroupReference:
    name: str
    start_tag: Tag
    end_tag: Tag


OrdMapEpsTrans = dict[State, list[tuple[Tag | None, State]]]
DblMapSymTrans = dict[State, dict[E, State]]
MapSymTrans = dict[tuple[State, E], set[State]]


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
    symbol_transitions: set[SymbolTransition[E]]  # ∆ - transitions on alphabet symbols
    epsilon_transitions: set[
        EpsilonTransition
    ]  # ∆ - optionally tagged ϵ-transitions with priority

    miltitags: set[Tag] = field(default_factory=set)
    named_groups_to_tags: dict[str, tuple[Tag, Tag]] = field(default_factory=dict)

    def dumps_dot(self) -> str:
        result = []
        result.append("digraph G {\n")
        result.append('node [label="", shape=circle, style=filled];\n\n')

        for state in self.states:
            if state == self.initial_state:
                result.append(f'n{state} [label="{state}", shape=doublecircle];\n')
            elif state == self.final_state:
                result.append(f'n{state} [label="{state}", shape=doublecircle];\n')
            else:
                result.append(f'n{state} [label="{state}"];\n')

        for source, priority, tag, target in self.epsilon_transitions:
            tag = "ε" if tag is None else tag
            result.append(
                f'n{source} -> n{target} [label="{priority}/{tag}"];\n'
            )  # , color=blue

        for source, symbol, target in self.symbol_transitions:
            result.append(
                f'n{source} -> n{target} [label="{symbol}"];\n'
            )  # , color=blue

        result.append("}\n")

        return "".join(result)

    def dump_dot(self, path: Path | str) -> Path:
        path = Path(path)
        path.write_text(self.dumps_dot(), encoding="utf-8")
        return path

    def to_dot_image(self, path: Path | str):
        path = self.dump_dot(path)
        os.system(f"dot -Tpng -Gdpi=300 {path} -o {path}.png")  # coz why not

    def get_mapped_symbol_transitions(self) -> MapSymTrans:
        mapped_sym = {}
        for q, s, p in self.symbol_transitions:
            val = mapped_sym.get((q, s), set())
            mapped_sym[(q, s)] = val | {p}
        return mapped_sym

    def get_double_mapped_symbol_transitions(self) -> DblMapSymTrans:
        mapped_sym = {}
        for q, s, p in self.symbol_transitions:
            val = mapped_sym[q] = mapped_sym.get(q, dict())
            val[s] = p
        return mapped_sym

    def get_ordered_mapped_epsilon_transitions(self) -> OrdMapEpsTrans:
        ordered_eps = {}
        for q, prior, tag, p in self.epsilon_transitions:
            val = ordered_eps.get(q, set())
            ordered_eps[q] = val | {(prior, tag, p)}
        ordered_eps = {
            q: [(it[1], it[2]) for it in sorted(trs, key=lambda x: x[0])]
            for q, trs in ordered_eps.items()
        }
        return ordered_eps

    def all_transitions(self, mapped_sym, states, symbol):
        syms = (mapped_sym.get(q, dict()).get(symbol) for q in states)
        return {it for it in syms if it is not None}

    def epsilon_reachable(self, ordered_eps, states: set[State]) -> set[State]:
        stack = deque(states)
        result = set()
        enqueued = set(stack)
        while stack:
            state = stack.pop()
            result.add(state)

            for tag, next_state in ordered_eps.get(state, []):
                if next_state not in enqueued:
                    stack.append(next_state)
                    enqueued.add(next_state)

        return result

    def run(self, word: str):
        ordered_eps = self.get_ordered_mapped_epsilon_transitions()
        mapped_sym = self.get_double_mapped_symbol_transitions()

        current_states = {self.initial_state}
        for symbol in word:
            full = self.epsilon_reachable(ordered_eps, current_states)
            current_states = self.all_transitions(mapped_sym, full, symbol)
        full = self.epsilon_reachable(ordered_eps, current_states)

        return self.final_state in full

    def as_simulatable(self):
        return SimulatableTNFA(
            self.get_ordered_mapped_epsilon_transitions(),
            self.get_double_mapped_symbol_transitions(),
            self.initial_state,
            self.final_state,
            self.named_groups_to_tags,
            dict(),
        )


SimTags = defaultdict[Tag, deque[int | None]]
SimConfs = dict[State, SimTags]


@dataclass
class SimulatableTNFA(Generic[E]):
    ordered_eps: OrdMapEpsTrans
    mapped_sym: DblMapSymTrans
    initial_state: State
    final_state: State
    named_groups_to_tags: dict[str, tuple[Tag, Tag]]
    confs: SimConfs

    @staticmethod
    def set_tag(registers: SimTags, tag: Tag | None, value: int):
        if tag is None:
            return

        if tag > 0:
            registers[tag].append(value)
        else:
            registers[-tag].append(None)

    def epsilon_closure(self, index: int) -> SimConfs:
        stack = deque(self.confs.items())
        enqueued = set(self.confs.keys())
        result = SimConfs()

        while stack:
            conf = stack.pop()

            tag_state_list = self.ordered_eps.get(conf[0], [])
            # print(conf, tag_state_list)
            for tag, next_state in tag_state_list:
                if next_state not in enqueued:
                    next_registers = deepcopy(conf[1])
                    self.set_tag(next_registers, tag, index)
                    stack.append((next_state, next_registers))
                    enqueued.add(next_state)

            if not tag_state_list:
                result[conf[0]] = conf[1]

        return result

    def step_on_symbol(self, symbol: str) -> SimConfs:
        result = SimConfs()
        for conf in self.confs.items():
            state = self.mapped_sym.get(conf[0], dict()).get(symbol)
            if state is not None:
                result[state] = conf[1]
        return result

    def gather_matches(self, word: str) -> dict[str, list[str]]:
        registers = self.confs[self.final_state]
        matches = defaultdict(list)
        for name, (start, end) in self.named_groups_to_tags.items():
            for start_id, end_id in zip(registers[start], registers[end]):
                if start_id is not None and end_id is not None:
                    matches[name].append(word[start_id: end_id])
                else:
                    matches[name].append(None)
        return dict(matches)

    def simulate(self, word: str):
        self.confs = {self.initial_state: defaultdict(deque)}

        for ind, symbol in enumerate(word):
            self.confs = self.epsilon_closure(ind)
            self.confs = self.step_on_symbol(symbol)
            if not self.confs:
                return None

        self.confs = self.epsilon_closure(len(word))

        if self.final_state not in self.confs:
            return None
        return self.gather_matches(word)


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


@dataclass
class AstLength(Visitor):
    cache: dict[ast.RE, int | None] = field(default_factory=dict)

    def visit(self, node):
        res = self.cache.get(node)
        if res is not None:
            return res
        res = super().visit(node)
        self.cache[node] = res
        return res

    def visit_Epsilon(self, node: ast.Epsilon):
        return 0

    def visit_Symbol(self, node: ast.Symbol):
        return 1

    def visit_Concat(self, node: ast.Concat):
        return sum(self.visit(child) for child in node.expressions)

    def visit_Or(self, node: ast.Or):
        parts = [self.visit(child) for child in node.expressions]
        unique = sum(1 for _ in iter_unique(parts))
        if unique != 1:
            return None
        return parts[0]



ALPHABET = set(string.printable)


class Ast2Tnfa(Visitor):
    """
    Augmented Thompson's construction for TNFA
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.next_state: State = 0
        self.next_tag: Tag = 0
        self.named_groups_to_tags = dict[str, tuple[Tag, Tag]]()
        self.found_named_groups = set[str]()

    def get_next_state(self):
        self.next_state += 1
        return self.next_state

    def get_next_tag(self):
        self.next_tag += 1
        return self.next_tag

    def to_nfa(self, node: ast.RE, initial_state: State = 0) -> TNFA:
        self.reset()
        self.next_state = initial_state
        tnfa = self.visit(node, initial_state)
        tnfa.named_groups_to_tags = self.named_groups_to_tags
        tnfa.miltitags = set()
        # tnfa.miltitags = set(tnfa.tags)
        return tnfa

    def negative_tags_from(self, tnfa: TNFA, state: State):
        states = [self.get_next_state() for _ in tnfa.tags] + [state]
        return TNFA(
            tnfa.alphabet,
            tnfa.tags,
            set(states),
            states[0],
            states[-1],
            set(),
            {
                EpsilonTransition(q, 1, -t, p)
                for q, t, p in zip(states, tnfa.tags, states[1:])
            },
        )

    def visit_Epsilon(self, node: ast.Epsilon, state: State):
        return TNFA(ALPHABET, set(), {state}, state, state, set(), set())

    def visit_Symbol(self, node: ast.Symbol, state: State):
        state2 = self.get_next_state()
        return TNFA(
            ALPHABET,
            set(),
            {state2, state},
            state2,
            state,
            {SymbolTransition(state2, node.value, state)},
            set(),
        )

    def visit_Tag(self, node: ast.Tag, state: State):
        state2 = self.get_next_state()
        return TNFA(
            ALPHABET,
            {node.value},
            {state2, state},
            state2,
            state,
            set(),
            {EpsilonTransition(state2, 1, node.value, state)},
        )

    def visit_Concat(self, node: ast.Concat, state: State):
        tnfas = []
        for expr in node.expressions[::-1]:
            tnfas.append(self.visit(expr, state))
            state = tnfas[-1].initial_state
        tnfas = tnfas[::-1]

        assert len(tnfas) > 0, "'concatenation' must have at least one expressions"

        alphabet = union((a.alphabet for a in tnfas), set())
        tags = union((a.tags for a in tnfas), set())
        states = union((a.states for a in tnfas), set())
        initial_state = tnfas[0].initial_state
        final_state = tnfas[-1].final_state
        symbol_transitions = union((a.symbol_transitions for a in tnfas), set())
        epsilon_transitions = union((a.epsilon_transitions for a in tnfas), set())

        return TNFA(
            alphabet,
            tags,
            states,
            initial_state,
            final_state,
            symbol_transitions,
            epsilon_transitions,
        )

    def or_with_two(self, lhs_node: ast.RE, rhs: TNFA, prioritize_lhs: bool = True):
        rhs_prime = self.negative_tags_from(rhs, rhs.final_state)
        lhs: TNFA = self.visit(lhs_node, rhs_prime.initial_state)
        lhs_prime = self.negative_tags_from(lhs, rhs.initial_state)

        tnfas = [lhs, lhs_prime, rhs, rhs_prime]

        start = self.get_next_state()

        alphabet = lhs.alphabet | rhs.alphabet
        tags = lhs.tags | rhs.tags
        states = union((a.states for a in tnfas), {start})
        initial_state = start
        final_state = rhs.final_state
        symbol_transitions = union((a.symbol_transitions for a in tnfas), set())
        epsilon_transitions = union(
            (a.epsilon_transitions for a in tnfas),
            {
                EpsilonTransition(
                    start, 1 if prioritize_lhs else 2, None, lhs.initial_state
                ),
                EpsilonTransition(
                    start, 2 if prioritize_lhs else 1, None, lhs_prime.initial_state
                ),
            },
        )

        return TNFA(
            alphabet,
            tags,
            states,
            initial_state,
            final_state,
            symbol_transitions,
            epsilon_transitions,
        )

    def visit_Or(self, node: ast.Or, state: State):
        assert len(node.expressions) > 0, "'or' must have at least one expressions"

        prev = self.visit(node.expressions[-1], state)
        for expr in node.expressions[:-1][::-1]:
            prev = self.or_with_two(expr, prev)

        return prev

    def visit_Repeat(self, node: ast.Repeat, state: State):
        return self.repeat_expr(node.expr, node.min, node.max, state)

    def repeat_expr(self, node: ast.RE, min: int, max: int | None, state: State):
        assert min >= 0, "'repeat' min must be non-negative"
        assert (
            max is None or max >= min
        ), "'repeat' max must be greater than or equal to min"

        if 1 < min:
            tnfa2 = self.repeat_expr(
                node, min - 1, max if max is None else max - 1, state
            )
            tnfa1 = self.visit(node, tnfa2.initial_state)
            return TNFA(
                tnfa1.alphabet | tnfa2.alphabet,
                tnfa1.tags | tnfa2.tags,
                tnfa1.states | tnfa2.states,
                tnfa1.initial_state,
                tnfa2.final_state,
                tnfa1.symbol_transitions | tnfa2.symbol_transitions,
                tnfa1.epsilon_transitions | tnfa2.epsilon_transitions,
            )

        if min == 0:
            tnfa = self.repeat_expr(node, min + 1, max, state)
            return self.or_with_two(ast.Epsilon(), tnfa, prioritize_lhs=False)

        if min == 1 and max == 1:
            return self.visit(node, state)

        if min == 1 and max is not None:
            end1 = self.get_next_state()

            tnfa2 = self.repeat_expr(node, 1, max - 1, state)
            tnfa1 = self.visit(node, end1)

            epsilon_transitions = tnfa1.epsilon_transitions | tnfa2.epsilon_transitions
            epsilon_transitions |= {
                EpsilonTransition(end1, 1, None, state),
                EpsilonTransition(end1, 2, None, tnfa2.initial_state),
            }
            return TNFA(
                tnfa1.alphabet | tnfa2.alphabet,
                tnfa1.tags | tnfa2.tags,
                tnfa1.states | tnfa2.states,
                tnfa1.initial_state,
                tnfa2.final_state,
                tnfa1.symbol_transitions | tnfa2.symbol_transitions,
                epsilon_transitions,
            )

        if min == 1:
            assert max is None, "'repeat' internal logic skipped a case"

            end = self.get_next_state()
            tnfa: TNFA = self.visit(node, end)
            states = tnfa.states | {state}
            epsilon_transitions = tnfa.epsilon_transitions | {
                EpsilonTransition(end, 1, None, tnfa.initial_state),
                EpsilonTransition(end, 2, None, state),
            }
            return TNFA(
                tnfa.alphabet,
                tnfa.tags,
                states,
                tnfa.initial_state,
                state,
                tnfa.symbol_transitions,
                epsilon_transitions,
            )

        assert False, "'repeat' did not match any case"

    def get_named_group_tags(self, name: str):
        if name in self.named_groups_to_tags:
            return self.named_groups_to_tags[name]

        start_tag = self.get_next_tag()
        end_tag = self.get_next_tag()
        self.named_groups_to_tags[name] = (start_tag, end_tag)
        return start_tag, end_tag

    def visit_NamedGroup(self, node: ast.NamedGroup, state: State):
        (start_tag, end_tag) = self.get_named_group_tags(node.name)
        self.found_named_groups.add(node.name)

        return self.visit(
            ast.Concat((ast.Tag(start_tag), node.expr, ast.Tag(end_tag))), state
        )

    def visit_NamedGroupReference(self, node: ast.NamedGroupReference, state: State):
        state2 = self.get_next_state()
        (start_tag, end_tag) = self.get_named_group_tags(node.name)
        sym = NamedGroupReference(node.name, start_tag, end_tag)
        return TNFA(
            ALPHABET | {sym},
            set(),
            {state2, state},
            state2,
            state,
            {SymbolTransition(state2, sym, state)},
            set(),
        )


_ast2tnfa = Ast2Tnfa()


def ast_to_tnfa(node: ast.RE) -> TNFA:
    return _ast2tnfa.to_nfa(node)
