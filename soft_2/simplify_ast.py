from __future__ import annotations

import classes as ast
from dataclasses import dataclass, field
from helpers import iter_unique


@dataclass
class AstLength(ast.Visitor):
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

    def visit_SymbolRanges(self, node: ast.SymbolRanges):
        return 1

    def visit_Concat(self, node: ast.Concat):
        result = 0
        for child in node.expressions:
            op = self.visit(child)
            if op is None:
                return None
            result += op
        return result

    def visit_Or(self, node: ast.Or):
        last_op = 0
        unique_count = 0
        for op in iter_unique(self.visit(e) for e in node.expressions):
            if op is None:
                return None
            unique_count += 1
            last_op = op

        if unique_count == 1:
            return last_op
        return None

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
class SimplifyAst(ast.Visitor):
    named_groups_to_tags: NGroup2Tags = field(default_factory=dict)
    next_tag: Tag = 0
    ast_length: AstLength = field(default_factory=AstLength)

    def visit_Epsilon(self, node: ast.Epsilon):
        return node

    def visit_Tag(self, node: ast.Tag):
        return node

    def visit_SymbolRanges(self, node: ast.SymbolRanges):
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
        ranges = []
        for child in node.expressions:
            child = self.visit(child)
            if isinstance(child, ast.Or):
                result.extend(child.expressions)
            elif isinstance(child, ast.SymbolRanges):
                ranges.append(child)
            else:
                result.append(child)
        result.extend(ast.SymbolRanges.merge(ranges))
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
            result = (ast.Tag(start_tag), expr)
        else:
            start_tag = self.get_next_tag()
            end_tag = self.get_next_tag()
            result = (ast.Tag(start_tag), expr, ast.Tag(end_tag))

        self.named_groups_to_tags[node.name] = (start_tag, end_tag)

        return ast.Concat(result)

    def visit_NamedGroupReference(self, node: ast.NamedGroupReference):
        return node
