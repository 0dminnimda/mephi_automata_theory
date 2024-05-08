from parser import parse
from dataclasses import asdict as _asdict
from pprint import pprint
from tnfa import ast_to_tnfa, TNFA
from tdfa import tnfa_to_tdfa, TDFA
import classes as ast
from pathlib import Path
import tdfa


_reported = []
_total_test_cases = 0


def asdict(obj, exclude=None):
    d = _asdict(obj)
    for k in exclude or set():
        d.pop(k, None)
    return d


def test_one_regex_tnfa(regex, cases, tnfa: TNFA):
    global _total_test_cases

    if DUMP_DOT:
        tnfa.dump_dot("tnfa_k.dot")
    simulatable_tnfa = tnfa.as_simulatable()

    for prompt, should_match, groups in cases:
        _total_test_cases += 1
        match_tnfa = simulatable_tnfa.simulate(prompt)
        if (match_tnfa is not None) != should_match:
            _reported.append(
                f"{prompt!r} should {'not ' if not should_match else ''}match {regex!r} in simulation [tNfa]"
            )
        else:
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


def test_one_regex_tdfa(regex, cases, tdfa: TDFA):
    global _total_test_cases

    if DUMP_DOT:
        tdfa.dump_dot("tdfa.dot")
    simulatable_tdfa = tdfa.as_simulatable()

    for prompt, should_match, groups in cases:
        _total_test_cases += 1
        match_tdfa = simulatable_tdfa.simulate(prompt)
        if (match_tdfa is not None) != should_match:
            _reported.append(
                f"{prompt!r} should {'not ' if not should_match else ''}match {regex!r} in simulation [tDfa]"
            )
        else:
            if match_tdfa is None:
                if groups != {}:
                    _reported.append(
                        f"{prompt!r} did not match but expected groups captured for {regex!r} [tDfa]"
                    )
            else:
                if groups != match_tdfa:
                    _reported.append(
                        f"{prompt!r} expected to match groups {groups}, but got {match_tdfa} for {regex!r} [tDfa]"
                    )


def test_one_regex(regex, cases):
    re = parse(regex)
    tnfa = ast_to_tnfa(re)
    tdfa = tnfa_to_tdfa(tnfa)

    if TEST_TNFA:
        test_one_regex_tnfa(regex, cases, tnfa)
    test_one_regex_tdfa(regex, cases, tdfa)



def test_regexes(data):
    for regex, cases in data.items():
        test_one_regex(regex, cases)

    for report in _reported:
        print(report)

    if _reported:
        print(f"!!! {len(_reported)} test cases out of {_total_test_cases} failed !!!")
        # exit(1)
    else:
        print(f"!!! All {_total_test_cases} test cases passed !!!")


data = {
    "b|((<gg>a)|%?%){2}?<gg>...": [
        ("", True, {"gg": [None]}),
        ("b", True, {"gg": [None]}),
        ("a", False, {}),
        ("?", False, {}),
        ("bb", False, {}),
        ("aa", True, {"gg": ["a", "a"]}),
        ("?a", True, {"gg": [None, "a"]}),
        ("a?", True, {"gg": ["a", None]}),
        ("??", True, {"gg": [None, None]}),
        ("aaa", True, {"gg": ["a", "a"]}),
        ("?aa", True, {"gg": [None, "a"]}),
        ("a?a", True, {"gg": ["a", None]}),
        ("??a", False, {}),
        ("?a?", False, {}),
        ("aaaa", True, {"gg": ["a", "a"]}),
        ("?aaa", True, {"gg": [None, "a"]}),
        ("a?aa", True, {"gg": ["a", None]}),
        ("??aa", False, {}),
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
    "a...[ab]b...": [
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
        ("aab", True, {"g2": ["b"], "g1": ["a", "a"]}),
        ("?a", False, {}),
        ("a?", False, {}),
        ("??", False, {}),
        ("??a", False, {}),
        ("?a?", False, {}),
        ("sdfsd", False, {}),
    ],
    "[ab]...[abcd][cd]...": [
        ("", False, {}),
        ("c", True, {}),
        ("a", True, {}),
        ("?", False, {}),
        ("aa", True, {}),
        ("cc", True, {}),
        ("aac", True, {}),
        ("?a", False, {}),
        ("a?", False, {}),
        ("??", False, {}),
        ("??a", False, {}),
        ("?a?", False, {}),
        ("d", True, {}),
        ("b", True, {}),
        ("?", False, {}),
        ("bb", True, {}),
        ("dd", True, {}),
        ("bbd", True, {}),
        ("?b", False, {}),
        ("b?", False, {}),
        ("??", False, {}),
        ("??b", False, {}),
        ("?b?", False, {}),
        ("sdfsd", False, {}),
    ],
    "[a-d]...[a-ho][e-h]...": [
        ("", False, {}),
        ("sdfsd", False, {}),
    ] + [
        (c, True, {}) for c in "abcdefgho"
    ] + [
        (c, True, {}) for c in "abcd"
    ] + [
        (c, True, {}) for c in "efgh"
    ] + [
        (c + "?", False, {}) for c in "abcdefgho"
    ] + [
        (c + "?", False, {}) for c in "abcd"
    ] + [
        (c + "?", False, {}) for c in "efgh"
    ] + [
        (c1 + c2, True, {}) for c1 in "abcdo" for c2 in "efgh"
    ] + [
        (c1 + c2, False, {}) for c1 in "efgh" for c2 in "abcd"
    ] + [
        (c1 + c2, True, {}) for c1 in "abcd" for c2 in "abcd"
    ] + [
        (c1 + c2, True, {}) for c1 in "efgh" for c2 in "efgh"
    ],
    "(<g1>[a-d])...(<g2>[a-ho])[e-h]...": [
        ("", False, {}),
        ("sdfsd", False, {}),
    ] + [
        (c, True, {"g1": [None], "g2": [c]}) for c in "abcd"  # XXX: why is it lazy here?
    ] + [
        (c, True, {"g1": [None], "g2": [c]}) for c in "efgho"
    ] + [
        (c + "?", False, {}) for c in "abcdefgho"
    ] + [
        (c + "?", False, {}) for c in "abcd"
    ] + [
        (c + "?", False, {}) for c in "efgh"
    ] + [
        (c1 + c2, True, {'g1': [c1], 'g2': [c2]}) for c1 in "abcd" for c2 in "efgh"
    ] + [
        ("o" + c2, True, {'g1': [None], 'g2': ["o"]}) for c2 in "efgh"
    ] + [
        (c1 + c2, False, {}) for c1 in "efgh" for c2 in "abcd"
    ] + [
        (c1 + c2, True, {'g1': [c1], 'g2': [c2]}) for c1 in "abcd" for c2 in "abcd"
    ] + [
        (c1 + c2, True, {'g1': [None], 'g2': [c1]}) for c1 in "efgh" for c2 in "efgh"
    ],
    "(<g1>[ab])...(<g2>[abcd])[cd]...": [
        ("", False, {}),
        ("d", True, {"g2": ["d"], "g1": [None]}),
        ("a", True, {"g2": ["a"], "g1": [None]}),
        ("?", False, {}),
        ("aa", True, {"g2": ["a"], "g1": ["a"]}),
        ("dd", True, {"g2": ["d"], "g1": [None]}),
        ("aad", True, {"g2": ["d"], "g1": ["a", "a"]}),
        ("?a", False, {}),
        ("a?", False, {}),
        ("??", False, {}),
        ("??a", False, {}),
        ("?a?", False, {}),
        ("c", True, {"g2": ["c"], "g1": [None]}),
        ("b", True, {"g2": ["b"], "g1": [None]}),
        ("?", False, {}),
        ("bb", True, {"g2": ["b"], "g1": ["b"]}),
        ("cc", True, {"g2": ["c"], "g1": [None]}),
        ("bbc", True, {"g2": ["c"], "g1": ["b", "b"]}),
        ("?b", False, {}),
        ("b?", False, {}),
        ("??", False, {}),
        ("??b", False, {}),
        ("?b?", False, {}),
        ("sdfsd", False, {}),
    ],
    "(<g1>a)...(<g2>[ab])b...": [
        ("", False, {}),
        ("b", True, {"g2": ["b"], "g1": [None]}),
        ("a", True, {"g2": ["a"], "g1": [None]}),
        ("?", False, {}),
        ("aa", True, {"g2": ["a"], "g1": ["a"]}),
        ("bb", True, {"g2": ["b"], "g1": [None]}),
        ("aab", True, {"g2": ["b"], "g1": ["a", "a"]}),
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
    "(a|b|c)+": [
        ("", False, {}),
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
    "(a|b|c|)+": [
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
    "((a?)...)+": [
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
    "%+%": [
        ("", False, {}),
        ("+", True, {}),
        ("++", False, {}),
        ("a", False, {}),
        ("b", False, {}),
        ("+a", False, {}),
        ("+b", False, {}),
    ],
    "[+]": [
        ("", False, {}),
        ("+", True, {}),
        ("++", False, {}),
        ("a", False, {}),
        ("b", False, {}),
        ("+a", False, {}),
        ("+b", False, {}),
    ],
    "[a-z]": [
        ("", False, {}),
    ]
    + [(chr(c), True, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c), False, {}) for c in range(ord("A"), ord("Z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("A"), ord("Z"))]
    + [(str(i), False, {}) for i in range(10)]
    + [(str(i)*2, False, {}) for i in range(10)],
    "[^a-z]": [
        ("", False, {}),
    ]
    + [(chr(c), False, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c), True, {}) for c in range(ord("A"), ord("Z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("A"), ord("Z"))]
    + [(str(i), True, {}) for i in range(10)]
    + [(str(i)*2, False, {}) for i in range(10)],
    "%w%": [
        ("", False, {}),
        (".", False, {}),
        (",", False, {}),
        (" ", False, {}),
    ]
    + [(chr(c), True, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c), True, {}) for c in range(ord("A"), ord("Z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("A"), ord("Z"))]
    + [(str(i), True, {}) for i in range(10)]
    + [(str(i)*2, False, {}) for i in range(10)],
    "%W%": [
        ("", False, {}),
        (".", True, {}),
        (",", True, {}),
        (" ", True, {}),
    ]
    + [(chr(c), False, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c), False, {}) for c in range(ord("A"), ord("Z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("A"), ord("Z"))]
    + [(str(i), False, {}) for i in range(10)]
    + [(str(i)*2, False, {}) for i in range(10)],
    "[a-bC-D]+": [
        ("", False, {}),
        ("a", True, {}),
        ("b", True, {}),
        ("C", True, {}),
        ("D", True, {}),
        ("ab", True, {}),
        ("bD", True, {}),
        ("Ca", True, {}),
        ("Db", True, {}),
        ("A", False, {}),
        ("B", False, {}),
        ("c", False, {}),
        ("d", False, {}),
        ("AC", False, {}),
        ("BD", False, {}),
        ("ca", False, {}),
        ("db", False, {}),
    ],
    "([^n]|n)+": [
        ("", False, {}),
        ("a", True, {}),
        ("b", True, {}),
        ("C", True, {}),
        ("D", True, {}),
        ("ab", True, {}),
        ("bD", True, {}),
        ("Ca", True, {}),
        ("Db", True, {}),
        ("A", True, {}),
        ("B", True, {}),
        ("c", True, {}),
        ("d", True, {}),
        ("AC", True, {}),
        ("BD", True, {}),
        ("ca", True, {}),
        ("db", True, {}),
    ],
    "*...": [
        ("", True, {}),
        ("a", True, {}),
        ("b", True, {}),
        ("C", True, {}),
        ("D", True, {}),
        ("ab", True, {}),
        ("bD", True, {}),
        ("Ca", True, {}),
        ("Db", True, {}),
        ("A", True, {}),
        ("B", True, {}),
        ("c", True, {}),
        ("d", True, {}),
        ("AC", True, {}),
        ("BD", True, {}),
        ("ca", True, {}),
        ("db", True, {}),
    ],
    "[<>]+": [
        ("", False, {}),
        ("a", False, {}),
        ("b", False, {}),
        ("<", True, {}),
        (">", True, {}),
        ("a>", False, {}),
        ("b<", False, {}),
        ("<a", False, {}),
        (">b", False, {}),
        ("><", True, {}),
        ("<<", True, {}),
        (">>", True, {}),
        ("<>", True, {}),
        ("<<>>", True, {}),
    ],
    "(a|b|e)|(a|b|c)": [
        ("", False, {}),
        ("a", True, {}),
        ("b", True, {}),
        ("c", True, {}),
        ("e", True, {}),
        ("aa", False, {}),
        ("bb", False, {}),
        ("cc", False, {}),
        ("ee", False, {}),
    ],
    "(a|b|e)|(a|b|c)|*": [
        ("", False, {}),
        ("a", True, {}),
        ("b", True, {}),
        ("c", True, {}),
        ("d", True, {}),
        ("e", True, {}),
        ("f", True, {}),
        ("aa", False, {}),
        ("bb", False, {}),
        ("cc", False, {}),
        ("dd", False, {}),
        ("ee", False, {}),
        ("ff", False, {}),
    ],
}


# fmt: off


def test_dfa0():
    # (<g1>a)*(<g2>a|<tag4>b)b*
    re = ast.Concat((
        ast.Repeat(ast.NamedGroup("g1", ast.make_symbol("a")), 0, None),
        ast.NamedGroup("g2", ast.Or((
            ast.make_symbol("a"),
            ast.Concat((ast.Tag(100), ast.make_symbol("b"))),
        ))),
        ast.Repeat(ast.make_symbol("b"), 0, None),
    ))
    tnfa = ast_to_tnfa(re)
    tdfa = tnfa_to_tdfa(tnfa)
    tdfa.dump_dot("tdfa.dot")


def test_dfa1():
    # ((<g1>a)*|(<g2>b)*)b?
    re = ast.Concat((
        ast.Or((
            ast.Repeat(ast.NamedGroup("g1", ast.make_symbol("a")), 0, None),
            ast.Repeat(ast.NamedGroup("g2", ast.make_symbol("b")), 0, None),
        )),
        ast.Repeat(ast.make_symbol("b"), 0, 1),
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
            ast.Repeat(ast.make_symbol("a"), 0, None),
            ast.Tag(100),
            ast.Repeat(ast.make_symbol("b"), 0, None),
        )),
        ast.Concat((
            ast.make_symbol("a"),
            ast.make_symbol("b"),
        )),
    ))
    tnfa = ast_to_tnfa(re)
    tdfa = tnfa_to_tdfa(tnfa)
    tdfa.dump_dot("tdfa.dot")


def test_dfa3():
    # b|((<o>a)|\?)f*
    # re = parse("b|((<gg>a)|%?%){2}?f...")
    # re = parse("((<gg>a)|%?%|<gg>){2}")
    # re = parse("(<a>a)...(<b>a|c)c...")
    re = parse("(%+%?(d)...)?(((%(%|%[%|%{%)?d...(%)%|%]%|%}%)?)...)")
    # re = ast.Or((
    #     ast.make_symbol("b"),
    #     ast.Concat((
    #         ast.Or((
    #             ast.NamedGroup("o", ast.make_symbol("a")),
    #             ast.make_symbol("?"),
    #         )),
    #         ast.Repeat(ast.make_symbol("f"), 0, None),
    #     ))
    # ))
    tnfa = ast_to_tnfa(re)
    tnfa.dump_dot("tnfa_k.dot")
    tdfa = tnfa_to_tdfa(tnfa)
    tdfa.dump_dot("tdfa.dot")


# fmt: on


TEST_TNFA = True
DUMP_DOT = True


if __name__ == "__main__":
    # test_dfa2()
    # test_dfa0()
    # test_dfa1()
    # test_dfa3()
    test_regexes(data)
    print("DONE")
