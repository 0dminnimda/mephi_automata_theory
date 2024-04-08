from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Sequence, Iterable, NamedTuple


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
    sumbol: E
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

    # symbol_transitions: dict[
    #     tuple[State, E], State
    # ]  # ∆ - transitions on alphabet symbols
    # epsilon_transitions: dict[
    #     State, set[tuple[Priority, Tag | None, State]]
    # ]  # ∆ - optionally tagged ϵ-transitions with priority

    named_groups_to_tags: dict[str, Tag] = field(default_factory=dict)

    # def shift_states_and_tags(self, state_offset: State, tag_offset: Tag):
    #     self.states = {s + state_offset for s in self.states}
    #     self.symbol_transitions = {
    #         (q + state_offset, a): p + state_offset
    #         for (q, a), p in self.symbol_transitions.items()
    #     }
    #     self.epsilon_transitions = {
    #         q + state_offset: {(i, t + tag_offset, p + state_offset) for (i, t, p) in ts}
    #         for q, ts in self.epsilon_transitions.items()
    #     }
    #     self.initial_states = {s + state_offset for s in self.initial_states}
    #     self.final_states = {s + state_offset for s in self.final_states}

    # @classmethod
    # def disjoin(cls, nfas: Sequence[TNFA[E]]):
    #     current_state = 0
    #     current_tag = 0
    #     for tnfa in nfas:
    #         current_state_save = current_state
    #         current_state += max(tnfa.states) + 1
    #         current_tag_save = current_tag
    #         current_tag += max(tnfa.tags) + 1
    #         tnfa.shift_states_and_tags(current_state_save, current_tag_save)

    def add_state(self):
        state = max(self.states) + 1
        self.states.add(state)
        return state

    # def transitions_into(self, state, symbol):
    #     return self.transitions.get((state, symbol), set())

    # def add_epsilon_transition(self, state1, state2):
    #     self.transitions[(state1, None)] = self.transitions_into(state1, None) | {state2}
    #     if state2 in self.final_states:
    #         self.final_states = self.final_states | {state1}

    # def all_transitions(self, states, symbol):
    #     return union_set(self.transitions_into(q, symbol) for q in states)

    # def epsilon_reachable(self, states) -> set[State]:
    #     epsilon_reachable = states
    #     while 1:
    #         states = self.all_transitions(states, None)
    #         if set.issubset(states, epsilon_reachable):
    #             break
    #         epsilon_reachable.update(states)
    #     return epsilon_reachable

    # def run(self, word):
    #     current_states = self.initial_states
    #     for symbol in word:
    #         full = self.epsilon_reachable(current_states)
    #         current_states = self.all_transitions(full, symbol)
    #     return not set.isdisjoint(current_states, self.final_states)


import classes as ast


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


import string


ALPHABET = set(string.printable)


@dataclass
class Ast2Tnfa(Visitor):
    next_state: State = 0
    next_tag: Tag = 0
    named_groups_to_tags: dict[str, tuple[Tag, Tag]] = field(default_factory=dict)

    def get_next_state(self):
        self.next_state += 1
        return self.next_state

    def get_next_tag(self):
        self.next_tag += 1
        return self.next_tag

    def to_nfa(self, node: ast.RE, initial_state: State = 0) -> TNFA:
        self.next_state = initial_state
        tnfa = self.visit(node, initial_state)
        tnfa.named_groups_to_tags = self.named_groups_to_tags
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
        for expr in node.expressions:
            tnfas.append(self.visit(expr, state))
            state = tnfas[-1].initial_state

        assert len(tnfas) > 0, "'concatenation' must have at least one expressions"

        tags = union((a.tags for a in tnfas), set())
        states = union((a.states for a in tnfas), set())
        initial_state = tnfas[0].initial_state
        final_state = tnfas[-1].final_state
        symbol_transitions = union((a.symbol_transitions for a in tnfas), set())
        epsilon_transitions = union((a.epsilon_transitions for a in tnfas), set())

        return TNFA(
            ALPHABET,
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
        lhs_prime = self.negative_tags_from(rhs, rhs.initial_state)

        tnfas = [lhs, lhs_prime, rhs, rhs_prime]

        start = self.get_next_state()

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
            ALPHABET,
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
        for expr in node.expressions[:-1]:
            prev = self.or_with_two(expr, prev)

        return prev

    def visit_AnyNumberOf(self, node: ast.AnyNumberOf, state: State):
        return self.repeat_expr(node.expr, 0, None, state)

    def visit_Maybe(self, node: ast.Maybe, state: State):
        return self.repeat_expr(node.expr, 0, 1, state)

    def visit_Repeat(self, node: ast.Repeat, state: State):
        return self.repeat_expr(node.expr, node.count, node.count, state)

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
                ALPHABET,
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
                ALPHABET,
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
                ALPHABET,
                tnfa.tags,
                states,
                tnfa.initial_state,
                state,
                tnfa.symbol_transitions,
                epsilon_transitions,
            )

        assert False, "'repeat' did not match any case"

    def visit_NamedGroup(self, node: ast.NamedGroup, state: State):
        start_tag = self.get_next_tag()
        end_tag = self.get_next_tag()
        self.named_groups_to_tags[node.name] = (start_tag, end_tag)

        return self.visit(
            ast.Concat((ast.Tag(start_tag), node.expr, ast.Tag(end_tag))), state
        )

    def visit_NamedGroupReference(self, node: ast.NamedGroupReference, state: State):
        state2 = self.get_next_state()
        (start_tag, end_tag) = self.named_groups_to_tags[node.name]
        sym = NamedGroupReference(node.name, start_tag, end_tag)
        return TNFA(
            ALPHABET,
            set(),
            {state2, state},
            state2,
            state,
            {SymbolTransition(state2, sym, state)},
            set(),
        )
