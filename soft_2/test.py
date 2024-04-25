from parser import parse
from dataclasses import asdict as _asdict
from pprint import pprint
from tnfa import ast_to_tnfa
from tdfa import tnfa_to_tdfa
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
    # print(regex)
    re = parse(regex)
    # print(re)
    tnfa = ast_to_tnfa(re)
    # tnfa.dump_dot("tnfa_k.dot")
    simulatable_tnfa = tnfa.as_simulatable()
    # pprint(asdict(tnfa, exclude={"alphabet"}), indent=4, width=200)
    # tdfa = tnfa_to_tdfa(tnfa)
    # pprint(asdict(tdfa, exclude={"alphabet"}), indent=4, width=200)

    tdfa = tnfa_to_tdfa(tnfa)
    # tdfa.dump_dot("tdfa.dot")
    simulatable_tdfa = tdfa.as_simulatable()

    for prompt, should_match, groups in cases:
        # if tnfa.run(prompt) != should_match:
        #     _reported.append(
        #         f"{prompt!r} should {'not'if not should_match else ''} match {regex!r} in run        [tNfa]"
        #     )
        match_tnfa = simulatable_tnfa.simulate(prompt)
        if (match_tnfa is not None) != should_match:
            _reported.append(
                f"{prompt!r} should {'not'if not should_match else ''} match {regex!r} in simulation [tNfa]"
            )

        if match_tnfa is None:
            if groups != {}:
                _reported.append(
                    f"{prompt!r} did not match but expected groups captured for {regex!r} [tNfa]"
                )
        else:
            if groups != match_tnfa:
                _reported.append(
                    f"{prompt!r} expected to match groups {groups}, but got {match_tnfa} for {regex!r} [tNfa]"
                )

        match_tdfa = simulatable_tdfa.simulate(prompt)
        if (match_tdfa is not None) != should_match:
            _reported.append(
                f"{prompt!r} should {'not'if not should_match else ''} match {regex!r} in simulation [tDfa]"
            )

        if match_tdfa is None:
            if groups != {}:
                _reported.append(
                    f"{prompt!r} did not match but expected groups captured for {regex!r} [tDfa]"
                )
        else:
            trankated_groups = {name: capture[-1:] for name, capture in groups.items()}
            if trankated_groups != match_tdfa:
                _reported.append(
                    f"{prompt!r} expected to match groups {trankated_groups}, but got {match_tdfa} for {regex!r} [tDfa]"
                )


def test_regexes(data):
    for regex, cases in data.items():
        test_one_regex(regex, cases)

    for report in _reported:
        print(report)

    if _reported:
        print(f"!!! {len(_reported)} test cases failed !!!")
        # exit(1)
    else:
        print("!!! All test cases passed !!!")


data = {
    "b|((<gg>a)|%?%){2}?<gg>...": [
        ("", True, {"gg": [None]}),
        ("b", True, {"gg": [None]}),
        ("a", False, {}),
        ("?", False, {}),
        ("aa", True, {"gg": ["a", "a"]}),
        ("?a", True, {"gg": [None, "a"]}),
        ("a?", True, {"gg": ["a", None]}),
        ("??", True, {"gg": [None, None]}),
        ("aaa", True, {}),
        ("?aa", True, {}),
        ("a?a", True, {}),
        ("??a", True, {}),
        ("?a?", False, {}),
        ("aaaa", True, {}),
        ("?aaa", True, {}),
        ("a?aa", True, {}),
        ("??aa", True, {}),
        ("sdfsd", False, {}),
    ],
    "b|(a|%?%){2}?": [
        ("", True, {}),
        ("b", True, {}),
        ("a", False, {}),
        ("?", False, {}),
        ("aa", True, {}),
        ("?a", True, {}),
        ("a?", True, {}),
        ("??", True, {}),
        ("??a", False, {}),
        ("?a?", False, {}),
        ("sdfsd", False, {}),
    ],
    "(a)": [
        ("", False, {}),
        ("a", True, {}),
        ("b", False, {}),
        ("ba", False, {}),
        ("ab", False, {}),
    ],
    "(abc)": [
        ("", False, {}),
        ("a", False, {}),
        ("b", False, {}),
        ("ba", False, {}),
        ("ab", False, {}),
        ("abc", True, {}),
        ("abcd", False, {}),
    ],
    "(<gg>a)": [
        ("", False, {}),
        ("a", True, {"gg": ["a"]}),
        ("b", False, {}),
        ("ba", False, {}),
        ("ab", False, {}),
    ],
    "(<gg>a){2}": [
        ("", False, {}),
        ("a", False, {}),
        ("b", False, {}),
        ("ba", False, {}),
        ("ab", False, {}),
        ("aa", True, {"gg": ["a", "a"]}),
        ("bb", False, {}),
        ("baa", False, {}),
        ("aba", False, {}),
        ("aaa", False, {}),
        ("bba", False, {}),
        ("bab", False, {}),
        ("abb", False, {}),
        ("aab", False, {}),
        ("bbb", False, {}),
    ],
    "b|((<gg>a)|%?%){2}?": [
        ("", True, {"gg": [None]}),
        ("b", True, {"gg": [None]}),
        ("a", False, {}),
        ("?", False, {}),
        ("aa", True, {"gg": ["a", "a"]}),
        ("?a", True, {"gg": [None, "a"]}),
        ("a?", True, {"gg": ["a", None]}),
        ("??", True, {"gg": [None, None]}),
        ("??a", False, {}),
        ("?a?", False, {}),
        ("sdfsd", False, {}),
    ],
    "a...(a|b)b...": [
        ("", False, {}),
        ("b", True, {}),
        ("a", True, {}),
        ("?", False, {}),
        ("aa", True, {}),
        ("bb", True, {}),
        ("aab", True, {}),
        ("?a", False, {}),
        ("a?", False, {}),
        ("??", False, {}),
        ("??a", False, {}),
        ("?a?", False, {}),
        ("sdfsd", False, {}),
    ],
    "(<g1>a)...(<g2>a|b)b...": [
        ("", False, {}),
        ("b", True, {"g2": ["b"], "g1": [None]}),
        ("a", True, {"g2": ["a"], "g1": [None]}),
        ("?", False, {}),
        ("aa", True, {"g2": ["a"], "g1": ["a"]}),
        ("bb", True, {"g2": ["b"], "g1": [None]}),
        ("aab", True, {"g2": ["a"], "g1": ["a"]}),
        ("?a", False, {}),
        ("a?", False, {}),
        ("??", False, {}),
        ("??a", False, {}),
        ("?a?", False, {}),
        ("sdfsd", False, {}),
    ],
    "(<h>a|b|c){3,}": [
        ("", False, {}),
        ("a", False, {}),
        ("b", False, {}),
        ("c", False, {}),
        ("ab", False, {}),
        ("ac", False, {}),
        ("bc", False, {}),
        ("abc", True, {"h": ["a", "b", "c"]}),
        ("abca", True, {"h": ["a", "b", "c", "a"]}),
        ("abcab", True, {"h": ["a", "b", "c", "a", "b"]}),
        ("abcaba", True, {"h": ["a", "b", "c", "a", "b", "a"]}),
        ("abcabac", True, {"h": ["a", "b", "c", "a", "b", "a", "c"]}),
        ("abcabacb", True, {"h": ["a", "b", "c", "a", "b", "a", "c", "b"]}),
        ("abcd", False, {}),
    ],
    "(a|b|c){,3}": [
        ("", True, {}),
        ("a", True, {}),
        ("b", True, {}),
        ("c", True, {}),
        ("ab", True, {}),
        ("ac", True, {}),
        ("bc", True, {}),
        ("abc", True, {}),
        ("abca", False, {}),
        ("abcab", False, {}),
        ("abcaba", False, {}),
        ("abcabac", False, {}),
        ("abcabacb", False, {}),
        ("abcd", False, {}),
    ],
    "(a|b|c){2,4}": [
        ("", False, {}),
        ("a", False, {}),
        ("b", False, {}),
        ("c", False, {}),
        ("ab", True, {}),
        ("ac", True, {}),
        ("bc", True, {}),
        ("abc", True, {}),
        ("abca", True, {}),
        ("abcab", False, {}),
        ("abcaba", False, {}),
        ("abcabac", False, {}),
        ("abcabacb", False, {}),
        ("abcd", False, {}),
    ],
    "(a|b|c){,}": [
        ("", True, {}),
        ("a", True, {}),
        ("b", True, {}),
        ("c", True, {}),
        ("ab", True, {}),
        ("ac", True, {}),
        ("bc", True, {}),
        ("abc", True, {}),
        ("abca", True, {}),
        ("abcab", True, {}),
        ("abcaba", True, {}),
        ("abcabac", True, {}),
        ("abcabacb", True, {}),
        ("abcd", False, {}),
    ],
    "(a|b|c|){3,}": [
        ("", True, {}),
        ("a", True, {}),
        ("b", True, {}),
        ("c", True, {}),
        ("ab", True, {}),
        ("ac", True, {}),
        ("bc", True, {}),
        ("abc", True, {}),
        ("abca", True, {}),
        ("abcab", True, {}),
        ("abcaba", True, {}),
        ("abcabac", True, {}),
        ("abcabacb", True, {}),
        ("abcd", False, {}),
    ],
    "(a?)...": [
        ("", True, {}),
        ("a", True, {}),
        ("b", False, {}),
        ("aa", True, {}),
        ("ab", False, {}),
        ("bb", False, {}),
        ("ba", False, {}),
        ("aaa", True, {}),
        ("aaaa", True, {}),
        ("aaaaa", True, {}),
    ],
}


# fmt: off


def test_dfa0():
    # (<g1>a)*(<g2>a|<tag4>b)b*
    re = ast.Concat((
        ast.Repeat(ast.NamedGroup("g1", ast.SymbolRange("a", "a")), 0, None),
        ast.NamedGroup("g2", ast.Or((
            ast.SymbolRange("a", "a"),
            ast.Concat((ast.Tag(100), ast.SymbolRange("b", "b"))),
        ))),
        ast.Repeat(ast.SymbolRange("b", "b"), 0, None),
    ))
    tnfa = ast_to_tnfa(re)
    tdfa = tnfa_to_tdfa(tnfa)
    tdfa.dump_dot("tdfa.dot")


def test_dfa1():
    # ((<g1>a)*|(<g2>b)*)b?
    re = ast.Concat((
        ast.Or((
            ast.Repeat(ast.NamedGroup("g1", ast.SymbolRange("a", "a")), 0, None),
            ast.Repeat(ast.NamedGroup("g2", ast.SymbolRange("b", "b")), 0, None),
        )),
        ast.Repeat(ast.SymbolRange("b", "b"), 0, 1),
    ))
    tnfa = ast_to_tnfa(re)
    tdfa = tnfa_to_tdfa(tnfa)
    tdfa.dump_dot("tdfa.dot")
    # print("tnfa", tnfa.as_simulatable().simulate("aaab"))
    # print("tdfa", tdfa.as_simulatable().simulate("aaab"))


def test_dfa2():
    # a*<tag>b*|ab
    re = ast.Or((
        ast.Concat((
            ast.Repeat(ast.SymbolRange("a", "a"), 0, None),
            ast.Tag(100),
            ast.Repeat(ast.SymbolRange("b", "b"), 0, None),
        )),
        ast.Concat((
            ast.SymbolRange("a", "a"),
            ast.SymbolRange("b", "b"),
        )),
    ))
    tnfa = ast_to_tnfa(re)
    tdfa = tnfa_to_tdfa(tnfa)
    tdfa.dump_dot("tdfa.dot")


# fmt: on


if __name__ == "__main__":
    # test_dfa2()
    # test_dfa0()
    test_dfa1()
    test_regexes(data)
    print("DONE")
