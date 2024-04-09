from parser import parse
from dataclasses import asdict as _asdict
from pprint import pprint
from tnfa import ast_to_tnfa
import classes as ast
from pathlib import Path
import tdfa


_reported = []


def asdict(obj, exclude=None):
    d = _asdict(obj)
    for k in exclude or set():
        d.pop(k, None)
    return d


def test_one_regex(regex, cases):
    re = parse(regex)
    # print(re)
    tnfa = ast_to_tnfa(re)
    # pprint(asdict(tnfa, exclude={"alphabet"}), indent=4, width=200)
    for prompt, should_match in cases:
        if tnfa.run(prompt) != should_match:
            _reported.append(f"{prompt!r} should {'not'if not should_match else ''} match {regex!r} in run")
        if tnfa.simulation(prompt) != should_match:
            _reported.append(f"{prompt!r} should {'not'if not should_match else ''} match {regex!r} in simulation")


def test_regexes(data):
    for regex, cases in data.items():
        test_one_regex(regex, cases)

    for report in _reported:
        print(report)

    if _reported:
        print(f"!!! {len(_reported)} test cases failed !!!")
        exit(1)
    else:
        print("!!! All test cases passed !!!")



data = {
    "b|((<gg>a)|%?%){2}?<gg>...": [
        ("", True),
        ("b", True),
        ("a", False),
        ("?", False),
        ("aa", True),
        ("?a", True),
        ("a?", True),
        ("??", True),
        ("aaa", True),
        ("?aa", True),
        ("a?a", True),
        ("??a", True),
        ("?a?", False),
        ("aaaa", True),
        ("?aaa", True),
        ("a?aa", True),
        ("??aa", True),
        ("sdfsd", False),
    ],
    "b|(a|%?%){2}?": [
        ("", True),
        ("b", True),
        ("a", False),
        ("?", False),
        ("aa", True),
        ("?a", True),
        ("a?", True),
        ("??", True),
        ("??a", False),
        ("?a?", False),
        ("sdfsd", False),
    ],
    "(a)": {
        ("", False),
        ("a", True),
        ("b", False),
        ("ba", False),
        ("ab", False),
    },
    "(abc)": {
        ("", False),
        ("a", False),
        ("b", False),
        ("ba", False),
        ("ab", False),
        ("abc", True),
        ("abcd", False),
    },
    "(<gg>a)": {
        ("", False),
        ("a", True),
        ("b", False),
        ("ba", False),
        ("ab", False),
    },
    "(<gg>a){2}": {
        ("", False),
        ("a", False),
        ("b", False),
        ("ba", False),
        ("ab", False),
        ("aa", True),
        ("bb", False),
        ("baa", False),
        ("aba", False),
        ("aaa", False),
        ("bba", False),
        ("bab", False),
        ("abb", False),
        ("aab", False),
        ("bbb", False),
    },
    "b|((<gg>a)|%?%){2}?": {
        ("", True),
        ("b", True),
        ("a", False),
        ("?", False),
        ("aa", True),
        ("?a", True),
        ("a?", True),
        ("??", True),
        ("??a", False),
        ("?a?", False),
        ("sdfsd", False),
    },
    "a*(a|b)b*": {
        ("", True),
        ("b", True),
        ("a", True),
        ("?", False),
        ("aa", True),
        ("bb", True),
        ("aab", True),
        ("?a", False),
        ("a?", False),
        ("??", False),
        ("??a", False),
        ("?a?", False),
        ("sdfsd", False),
    },
    "(<g1>a)*(<g2>a|b)b*": {
        ("", True),
        ("b", True),
        ("a", True),
        ("?", False),
        ("aa", True),
        ("bb", True),
        ("aab", True),
        ("?a", False),
        ("a?", False),
        ("??", False),
        ("??a", False),
        ("?a?", False),
        ("sdfsd", False),
    },
}


def test_dfa0():
    # (<g1>a)*(<g2>a|<tag4>b)b*
    re = ast.Concat((
        ast.AnyNumberOf(ast.NamedGroup("g1", ast.Symbol("a"))),
        ast.NamedGroup("g2", ast.Or((
            ast.Symbol("a"),
            ast.Concat((ast.Tag(100), ast.Symbol("b"))),
        ))),
        ast.AnyNumberOf(ast.Symbol("b")),
    ))
    tnfa = ast_to_tnfa(re)
    pprint(asdict(tnfa, exclude={"alphabet"}), indent=4, width=200)
    # tnfa.to_dot_image("tnfa.dot")
    tnfa.dump_dot("tnfa0.dot")
    tdfa.determinization(tnfa)


def test_dfa1():
    # ((<g1>a)*|(<g2>b)*)b?
    re = ast.Concat((
        ast.Or((
            ast.AnyNumberOf(ast.NamedGroup("g1", ast.Symbol("a"))),
            ast.AnyNumberOf(ast.NamedGroup("g2", ast.Symbol("b"))),
        )),
        ast.Maybe(ast.Symbol("b")),
    ))
    tnfa = ast_to_tnfa(re)
    pprint(asdict(tnfa, exclude={"alphabet"}), indent=4, width=200)
    # tnfa.to_dot_image("tnfa.dot")
    tnfa.dump_dot("tnfa1.dot")
    tdfa.determinization(tnfa)


def test_dfa2():
    # a*<tag>b*|ab
    re = ast.Or((
        ast.Concat((
            ast.AnyNumberOf(ast.Symbol("a")),
            ast.Tag(100),
            ast.AnyNumberOf(ast.Symbol("b")),
        )),
        ast.Concat((
            ast.Symbol("a"),
            ast.Symbol("b"),
        )),
    ))
    tnfa = ast_to_tnfa(re)
    pprint(asdict(tnfa, exclude={"alphabet"}), indent=4, width=200)
    # tnfa.to_dot_image("tnfa.dot")
    tnfa.dump_dot("tnfa2.dot")
    tdfa.determinization(tnfa)


if __name__ == "__main__":
    # test_dfa2()
    test_dfa0()
    # test_dfa1()
    test_regexes(data)
