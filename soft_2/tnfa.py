from __future__ import annotations

import os
import string
import classes as ast
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, Generic, Sequence, NamedTuple
from collections import deque, defaultdict
from pathlib import Path
from simplify_ast import Visitor, Tag, FixedTag, AnyTag, NGroup2Tags, SimplifyAst

from copy import deepcopy


State = int
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


@dataclass(frozen=True)
class NamedGroupReference:
    start_tag: AnyTag
    end_tag: AnyTag


Matcher = ast.SymbolRange | NamedGroupReference


def dump_tag(tag: AnyTag) -> str:
    if isinstance(tag, FixedTag):
        return f"({tag.origin})+{tag.offset}"
    else:
        return f"({tag})"


def dump_matcher(matcher: Matcher) -> str:
    if isinstance(matcher, ast.SymbolRange):
        if matcher.start == matcher.end:
            return f"{matcher.start}"
        else:
            return f"[{matcher.start}-{matcher.end}]"
    else:
        return f"ref<{dump_tag(matcher.start_tag)}: {dump_tag(matcher.end_tag)}>"


class EpsilonTransition(NamedTuple):
    source: State
    priority: Priority
    tag: Tag | None
    target: State


class SymbolTransition(NamedTuple):
    source: State
    symbol: Matcher
    target: State


OrdMapEpsTrans = dict[State, list[tuple[Tag | None, State]]]
DblMapSymTrans = dict[State, dict[Matcher, State]]


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
    symbol_transitions: set[SymbolTransition]  # ∆ - transitions on alphabet symbols
    epsilon_transitions: set[
        EpsilonTransition
    ]  # ∆ - optionally tagged ϵ-transitions with priority

    multitags: set[Tag] = field(default_factory=set)
    named_groups_to_tags: NGroup2Tags = field(default_factory=NGroup2Tags)

    def dumps_dot(self) -> str:
        result = []
        result.append("digraph G {\n")
        result.append("rankdir=LR\n")
        result.append('node [label="", shape=circle, style=filled, fontname=Courier];\n')
        result.append("edge[arrowhead=vee fontname=Courier]\n")
        result.append("\n")

        for source, priority, tag, target in self.epsilon_transitions:
            tag = "ε" if tag is None else tag
            result.append(
                f'n{source} -> n{target} [label="{priority}/{tag}"];\n'
            )  # , color=blue

        for source, matcher, target in self.symbol_transitions:
            result.append(
                f'n{source} -> n{target} [label="{dump_matcher(matcher)}"];\n'
            )  # , color=blue

        for state in self.states:
            if state == self.initial_state:
                result.append(f'n{state} [label="{state}", shape=doublecircle];\n')
            elif state == self.final_state:
                result.append(f'n{state} [label="{state}", shape=doublecircle];\n')
            else:
                result.append(f'n{state} [label="{state}"];\n')

        result.append("}\n")

        return "".join(result)

    def dump_dot(self, path: Path | str) -> Path:
        path = Path(path)
        path.write_text(self.dumps_dot(), encoding="utf-8")
        return path

    def to_dot_image(self, path: Path | str):
        path = self.dump_dot(path)
        os.system(f"dot -Tpng -Gdpi=300 {path} -o {path}.png")  # coz why not

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

@dataclass
class SimConf:
    registers: SimTags = field(default_factory=lambda: defaultdict(deque))
    index: int = 0

    def set_tag(self, tag: Tag | None, value: int):
        if tag is None:
            return

        if tag > 0:
            self.registers[tag].append(value)
        else:
            self.registers[-tag].append(None)


SimConfs = dict[State, SimConf]


@dataclass
class SimulatableTNFA(Generic[E]):
    ordered_eps: OrdMapEpsTrans
    mapped_sym: DblMapSymTrans
    initial_state: State
    final_state: State
    named_groups_to_tags: NGroup2Tags
    confs: SimConfs

    def epsilon_closure(self, confs) -> SimConfs:
        stack = deque(confs.items())
        enqueued = set(confs.keys())
        result = SimConfs()

        while stack:
            conf_state, conf = stack.pop()

            tag_state_list = self.ordered_eps.get(conf_state, [])
            # print(conf, tag_state_list)
            for tag, next_state in tag_state_list:
                if next_state not in enqueued:
                    next_conf = deepcopy(conf)
                    next_conf.set_tag(tag, conf.index)
                    stack.append((next_state, next_conf))
                    enqueued.add(next_state)

            if not tag_state_list:
                result[conf_state] = conf

        return result

    def run_matcher(self, matcher: Matcher, word: str, index: int) -> int | None:
        if isinstance(matcher, ast.SymbolRange):
            if matcher.start <= word[index] <= matcher.end:
                return index + 1
            else:
                return None
        else:
            return None
            # raise NotImplementedError("groups")

    def step_on_symbol(self, word: str) -> SimConfs:
        result = SimConfs()
        for conf_state, conf in self.confs.items():
            assert conf.index < len(word)
            for matcher, state in self.mapped_sym.get(conf_state, dict()).items():
                index = self.run_matcher(matcher, word, conf.index)
                if index is not None:
                    result[state] = SimConf(conf.registers, index)
        return result

    def separate_finished(self, word: str) -> tuple[SimConfs, SimConfs]:
        finished = SimConfs()
        not_finished = SimConfs()
        for conf_state, conf in self.confs.items():
            if conf.index < len(word):
                not_finished[conf_state] = conf
            else:
                finished[conf_state] = conf
        return finished, not_finished

    def get_register_storage(self, regs: SimTags, tag: AnyTag) -> tuple[Sequence[int | None], int]:
        if isinstance(tag, FixedTag):
            return regs[tag.origin], tag.offset
        else:
            return regs[tag], 0

    def gather_matches(self, conf: SimConf, word: str) -> dict[str, list[str]]:
        registers = conf.registers
        matches = defaultdict(list)
        for name, (start, end) in self.named_groups_to_tags.items():
            start_indices, start_offset = self.get_register_storage(registers, start)
            end_indices, end_offset = self.get_register_storage(registers, end)
            for start_id, end_id in zip(start_indices, end_indices):
                if start_id is not None and end_id is not None:
                    matches[name].append(word[start_id + start_offset: end_id + end_offset])
                else:
                    matches[name].append(None)
        return dict(matches)

    def simulate(self, word: str):
        self.confs = {self.initial_state: SimConf()}
        finished = dict()

        while self.confs:
            self.confs = self.epsilon_closure(self.confs)
            finished, self.confs = self.separate_finished(word)
            finished = self.epsilon_closure(finished)
            good = finished.get(self.final_state)
            if good is not None:
                return self.gather_matches(good, word)

            self.confs = self.step_on_symbol(word)

        return None



ALPHABET = set(string.printable)



class Ast2Tnfa(Visitor):
    """
    Augmented Thompson's construction for TNFA
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.next_state: State = 0
        self.expand_ast = SimplifyAst()

    def get_next_state(self):
        self.next_state += 1
        return self.next_state

    def to_nfa(self, node: ast.RE) -> TNFA:
        self.reset()
        node = self.expand_ast.visit(node)
        tnfa = self.visit(node, self.next_state)
        tnfa.named_groups_to_tags = self.expand_ast.named_groups_to_tags
        # tnfa.multitags = set()
        tnfa.multitags = set(tnfa.tags)
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

    def visit_SymbolRange(self, node: ast.SymbolRange, state: State):
        state2 = self.get_next_state()
        return TNFA(
            ALPHABET,
            set(),
            {state2, state},
            state2,
            state,
            {SymbolTransition(state2, node, state)},
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

    def visit_NamedGroup(self, node: ast.NamedGroup, state: State):
        assert False, "Ast2Tnfa only accepts expanded ast (got NamedGroup)"

    def visit_NamedGroupReference(self, node: ast.NamedGroupReference, state: State):
        state2 = self.get_next_state()
        tags = self.expand_ast.named_groups_to_tags.get(node.name)
        if tags is None:
            raise ValueError(f"Named group {node.name} not found, but referenced")

        start_tag, end_tag = tags
        sym = NamedGroupReference(start_tag, end_tag)
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
