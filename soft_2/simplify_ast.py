from __future__ import annotations

import classes as ast
from dataclasses import dataclass, field
from parser import iter_unique


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
    groups: dict[str, int | None] = field(default_factory=dict)

    def visit(self, node):
        res = self.cache.get(node)
        if res is not None:
            return res
        res = super().visit(node)
        self.cache[node] = res
        return res

    def visit_Epsilon(self, node: ast.Epsilon):
        return 0

    def visit_Tag(self, node: ast.Tag):
        return 0

    def visit_SymbolRange(self, node: ast.SymbolRange):
        return 1

    def visit_Concat(self, node: ast.Concat):
        return sum(self.visit(child) for child in node.expressions)

    def visit_Or(self, node: ast.Or):
        parts = [self.visit(child) for child in node.expressions]
        unique = sum(1 for _ in iter_unique(parts))
        if unique != 1:
            return None
        return parts[0]

    def visit_Repeat(self, node: ast.Repeat):
        if node.min == node.max:
            return self.visit(node.expr) * node.min
        return None

    def visit_NamedGroup(self, node: ast.NamedGroup):
        res = self.visit(node.expr)
        self.groups[node.name] = res
        return res

    def visit_NamedGroupReference(self, node: ast.NamedGroupReference):
        return self.groups.get(node.name)


Tag = int


@dataclass(frozen=True)
class FixedTag:
    origin: Tag
    offset: int


AnyTag = Tag | FixedTag


NGroup2Tags = dict[str, tuple[AnyTag, AnyTag]]


@dataclass
class SimplifyAst(Visitor):
    named_groups_to_tags: NGroup2Tags = field(default_factory=dict)
    next_tag: Tag = 0
    ast_length: AstLength = field(default_factory=AstLength)

    def visit_Epsilon(self, node: ast.Epsilon):
        return node

    def visit_Tag(self, node: ast.Tag):
        return node

    def visit_SymbolRange(self, node: ast.SymbolRange):
        return node

    def visit_Concat(self, node: ast.Concat):
        result = []
        for child in node.expressions:
            child = self.visit(child)
            if isinstance(child, ast.Concat):
                result.extend(child.expressions)
            else:
                result.append(child)
        return ast.Concat(tuple(result))

    def visit_Or(self, node: ast.Or):
        result = []
        for child in node.expressions:
            child = self.visit(child)
            if isinstance(child, ast.Or):
                result.extend(child.expressions)
            else:
                result.append(child)
        return ast.Or(tuple(result))

    def visit_Repeat(self, node: ast.Repeat):
        return ast.Repeat(self.visit(node.expr), node.min, node.max)

    def get_next_tag(self):
        self.next_tag += 1
        return self.next_tag

    def visit_NamedGroup(self, node: ast.NamedGroup):
        if node.name in self.named_groups_to_tags:
            raise ValueError(f"group '{node.name}' is defined more than once")

        expr = self.visit(node.expr)

        length = self.ast_length.visit(expr)
        if isinstance(length, int):
            start_tag = self.get_next_tag()
            end_tag = FixedTag(start_tag, length)
            result = (ast.Tag(start_tag), node.expr)
        else:
            start_tag = self.get_next_tag()
            end_tag = self.get_next_tag()
            result = (ast.Tag(start_tag), node.expr, ast.Tag(end_tag))

        self.named_groups_to_tags[node.name] = (start_tag, end_tag)

        return ast.Concat(result)

    def visit_NamedGroupReference(self, node: ast.NamedGroupReference):
        return node
