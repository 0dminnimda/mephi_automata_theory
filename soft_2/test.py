from parser import parse
from dataclasses import asdict as _asdict
from pprint import pprint
from tnfa import ast_to_tnfa, TNFA
from tdfa import tnfa_to_tdfa, TDFA
import classes as ast
from library import Pattern
from pathlib import Path
import tdfa


_reported = []
_total_test_cases = 0


def asdict(obj, exclude=None):
    d = _asdict(obj)
    for k in exclude or set():
        d.pop(k, None)
    return d


def test_one_regex_full_match_tnfa(regex, cases, tnfa: TNFA):
    global _total_test_cases
    _total_test_cases += len(cases)

    sim = tnfa.as_simulatable()

    for prompt, should_match, groups in cases:
        match = sim.simulate(prompt)
        if (match is not None) != should_match:
            _reported.append(
                f"{prompt!r} should {'not ' if not should_match else ''}match {regex!r} [tNfa]"
            )
        else:
            if match is None:
                if groups != {}:
                    _reported.append(
                        f"{prompt!r} did not match but expected groups captured for {regex!r} [tNfa]"
                    )
            else:
                if groups != match:
                    _reported.append(
                        f"{prompt!r} expected to match groups {groups}, but got {match} for {regex!r} [tNfa]"
                    )


def test_one_regex_full_match_tdfa(regex, cases, tdfa: TDFA):
    global _total_test_cases
    _total_test_cases += len(cases)

    pattern = Pattern(tdfa.as_simulatable())
    rere = pattern.restore_regex_via_k_path()

    if rere is not None:
        _total_test_cases += len(cases)
        pattern_2 = Pattern(rere)
        if DUMP_DOT:
            pattern_2._simulatable.to_partial_tdfa().dump_dot("tdfa_restore.dot")
    else:
        pattern_2 = None

    pattern_3 = pattern.reverse()
    if pattern_3 is not None:
        _total_test_cases += len(cases)
        if DUMP_DOT:
            pattern_3._simulatable.to_partial_tdfa().dump_dot("tdfa_reverse.dot")

    pattern_4 = pattern.complement()
    if pattern_4 is not None:
        _total_test_cases += len(cases)
        if DUMP_DOT:
            pattern_4._simulatable.to_partial_tdfa().dump_dot("tdfa_comp.dot")

    for prompt, should_match, groups in cases:
        match = pattern.match(prompt)
        if (match is not None) != should_match:
            _reported.append(
                f"{prompt!r} should {'not ' if not should_match else ''}match {regex!r} [tDfa]"
            )
        else:
            if match is None:
                if groups != {}:
                    _reported.append(
                        f"{prompt!r} did not match but expected groups captured for {regex!r} [tDfa]"
                    )
            else:
                if groups != match.groups:
                    _reported.append(
                        f"{prompt!r} expected to match groups {groups}, but got {match.groups} for {regex!r} [tDfa]"
                    )

        if pattern_2 is not None:
            match2 = pattern_2.match(prompt)

            if (match2 is not None) != should_match:
                _reported.append(
                    f"{prompt!r} should {'not ' if not should_match else ''}match {rere!r} (from {regex!r}) [tDfa]"
                )

        if pattern_3 is not None:
            prompt3 = prompt[::-1]
            match3 = pattern_3.match(prompt3)

            if (match3 is not None) != should_match:
                _reported.append(
                    f"{prompt3!r} should {'not ' if not should_match else ''}match reversed({regex!r}) [tDfa]"
                )

        if pattern_4 is not None:
            match4 = pattern_4.match(prompt)
            should_match4 = not should_match

            if (match4 is not None) != should_match4:
                _reported.append(
                    f"{prompt!r} should {'not ' if not should_match4 else ''}match complemented({regex!r}) [tDfa]"
                )


def test_one_regex_full_match(regex, cases):
    re = parse(regex)
    tnfa = ast_to_tnfa(re)
    tdfa = tnfa_to_tdfa(tnfa)

    if DUMP_DOT:
        tnfa.dump_dot("tnfa.dot")
    if DUMP_DOT:
        tdfa.dump_dot("tdfa.dot")

    if TEST_TNFA:
        test_one_regex_full_match_tnfa(regex, cases, tnfa)
    test_one_regex_full_match_tdfa(regex, cases, tdfa)


def test_one_regex_find_all(regex, cases):
    global _total_test_cases
    _total_test_cases += len(cases)

    pattern = Pattern(regex)

    for prompt, result in cases:
        actual = pattern.findall(prompt)
        result_all = [s for s, _ in result]
        if actual != result_all:
            _reported.append(
                f"{prompt!r} expected to to return {result_all}, but got {actual} for {regex!r} [tDfa]"
            )

        actual = list(pattern.finditer(prompt))
        actual_iter = [(match.string, match.groups) for match in actual]
        if actual_iter != result:
            _reported.append(
                f"{prompt!r} expected to to return {result}, but got {actual_iter} for {regex!r} [tDfa]"
            )


def test_regexes(data_full_match, data_find_all):
    for regex, cases in data_full_match.items():
        test_one_regex_full_match(regex, cases)

    for regex, cases in data_find_all.items():
        test_one_regex_find_all(regex, cases)

    for report in _reported:
        print(report)

    if _reported:
        print(f"!!! {len(_reported)} test cases out of {_total_test_cases} failed !!!")
        # exit(1)
    else:
        print(f"!!! All {_total_test_cases} test cases passed !!!")


def make_giant_email_pattern():
    local_name = r"[a-z0-9]+"

    dotted_local = f"{local_name}(.{local_name})..."

    quoted_local = '"([\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-[%]%-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])..."'

    local = f"({dotted_local}|{quoted_local})"

    dnss_label = "[a-z0-9]([a-z0-9%-%]...[a-z0-9])?"

    dotted_domain = rf"({dnss_label}.)+{dnss_label}"

    # ip_digit = r"25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?"
    # ipv4 = rf"(({ip_digit})\.){{3}}({ip_digit})"

    ipv4 = "((25[0-5]|(2[0-4]|1%d%|[1-9]|)%d%).?){4}"

    ipv6_segment = "[0-9a-fA-F]{1,4}"
    ipv6 = (
        f"({ipv6_segment}:){{7,7}}{ipv6_segment}|"
        f"({ipv6_segment}:){{1,7}}:|"
        f"({ipv6_segment}:){{1,6}}:{ipv6_segment}|"
        f"({ipv6_segment}:){{1,5}}(:{ipv6_segment}){{1,2}}|"
        f"({ipv6_segment}:){{1,4}}(:{ipv6_segment}){{1,3}}|"
        f"({ipv6_segment}:){{1,3}}(:{ipv6_segment}){{1,4}}|"
        f"({ipv6_segment}:){{1,2}}(:{ipv6_segment}){{1,5}}|"
        f"{ipv6_segment}:((:{ipv6_segment}){{1,6}})|"
        f":((:{ipv6_segment}){{1,7}}|:)|"
        f"fe80:(:{ipv6_segment}){{0,4}}%[0-9a-zA-Z]{{1,}}|"
        f"::(ffff(:0{{1,4}}){{0,1}}:){{0,1}}{ipv4}|"
        f"({ipv6_segment}:){{1,4}}:{ipv4}"
    )

    ip_wtfkw = "([a-z0-9%-%]...[a-z0-9]:([\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)"

    ip_domain = f"%[%((IP(v4)?:)?{ipv4}|(IPv6:)?{ipv6}|{ip_wtfkw})%]%"

    domain = f"({dotted_domain}|{ip_domain})"

    regular = f"(<local>{local})%s%...@%s%...(<domain>{domain})"

    return regular


GIANT_EMAIL_PATTERN = make_giant_email_pattern()


def make_email_pattern():
    local_name = r"[a-z0-9]+"
    local = f"{local_name}(.{local_name})..."

    dnss_label = "[a-z0-9]([a-z0-9%-%]...[a-z0-9])?"
    domain = rf"({dnss_label}.)+{dnss_label}"

    return f"(<local>{local})%s%...@%s%...(<domain>{domain})"


EMAIL_PATTERN = make_email_pattern()


data_full_match = {
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
    "(<g1>a...)[ab](<g2>b...)": [
        ("", False, {}),
        ("b", True, {"g1": [""], "g2": [""]}),
        ("a", True, {"g1": [""], "g2": [""]}),
        ("?", False, {}),
        ("aa", True, {"g1": ["a"], "g2": [""]}),
        ("bb", True, {"g1": [""], "g2": ["b"]}),
        ("aab", True, {"g1": ["aa"], "g2": [""]}),
        ("abb", True, {"g1": ["a"], "g2": ["b"]}),
        ("aaab", True, {"g1": ["aaa"], "g2": [""]}),
        ("aabb", True, {"g1": ["aa"], "g2": ["b"]}),
        ("abbb", True, {"g1": ["a"], "g2": ["bb"]}),
        ("aaa", True, {"g1": ["aa"], "g2": [""]}),
        ("bbb", True, {"g1": [""], "g2": ["bb"]}),
        ("?a", False, {}),
        ("a?", False, {}),
        ("??", False, {}),
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
    ]
    + [(c, True, {}) for c in "abcdefgho"]
    + [(c, True, {}) for c in "abcd"]
    + [(c, True, {}) for c in "efgh"]
    + [(c + "?", False, {}) for c in "abcdefgho"]
    + [(c + "?", False, {}) for c in "abcd"]
    + [(c + "?", False, {}) for c in "efgh"]
    + [(c1 + c2, True, {}) for c1 in "abcdo" for c2 in "efgh"]
    + [(c1 + c2, False, {}) for c1 in "efgh" for c2 in "abcd"]
    + [(c1 + c2, True, {}) for c1 in "abcd" for c2 in "abcd"]
    + [(c1 + c2, True, {}) for c1 in "efgh" for c2 in "efgh"],
    "(<g1>[a-d])...(<g2>[a-ho])[e-h]...": [
        ("", False, {}),
        ("sdfsd", False, {}),
    ]
    + [(c, True, {"g1": [None], "g2": [c]}) for c in "abcd"]
    + [(c, True, {"g1": [None], "g2": [c]}) for c in "efgho"]
    + [(c + "?", False, {}) for c in "abcdefgho"]
    + [(c + "?", False, {}) for c in "abcd"]
    + [(c + "?", False, {}) for c in "efgh"]
    + [(c1 + c2, True, {"g1": [c1], "g2": [c2]}) for c1 in "abcd" for c2 in "efgh"]
    + [("o" + c2, True, {"g1": [None], "g2": ["o"]}) for c2 in "efgh"]
    + [(c1 + c2, False, {}) for c1 in "efgh" for c2 in "abcd"]
    + [(c1 + c2, True, {"g1": [c1], "g2": [c2]}) for c1 in "abcd" for c2 in "abcd"]
    + [(c1 + c2, True, {"g1": [None], "g2": [c1]}) for c1 in "efgh" for c2 in "efgh"],
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
    + [(str(i) * 2, False, {}) for i in range(10)],
    "[^a-z]": [
        ("", False, {}),
    ]
    + [(chr(c), False, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("a"), ord("z"))]
    + [(chr(c), True, {}) for c in range(ord("A"), ord("Z"))]
    + [(chr(c) * 2, False, {}) for c in range(ord("A"), ord("Z"))]
    + [(str(i), True, {}) for i in range(10)]
    + [(str(i) * 2, False, {}) for i in range(10)],
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
    + [(str(i) * 2, False, {}) for i in range(10)],
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
    + [(str(i) * 2, False, {}) for i in range(10)],
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
    "(<ou>ab(<in>[c-f]))...": [
        ("", True, {"in": [None], "ou": [None]}),
        ("b", False, {}),
        ("a", False, {}),
        ("aa", False, {}),
        ("bb", False, {}),
        ("aab", False, {}),
        ("abc", True, {"ou": ["abc"], "in": ["c"]}),
        ("abcabdabf", True, {"ou": ["abc", "abd", "abf"], "in": ["c", "d", "f"]}),
        ("sdfsd", False, {}),
    ],
    "(<ou>%(%<ou>%)%)+": [
        ("", False, {}),
        ("b", False, {}),
        ("a", False, {}),
        ("()", False, {}),
        ("()()", False, {}),
        ("()(())", False, {}),
        ("()(())(())", False, {}),
        ("()(())((()))", False, {}),
    ],
    "(<ou>%(%%)%)<ou>+": [
        ("", False, {}),
        ("b", False, {}),
        ("a", False, {}),
        ("()", False, {}),
        ("()()", True, {"ou": ["()"]}),
        ("()()()", True, {"ou": ["()"]}),
        ("()(())", False, {}),
        ("()(())((()))", False, {}),
    ],
    "(<ou>ab(<in>[c-f]))...(|<ou>|<in>)...": [
        ("", True, {"in": [None], "ou": [None]}),
        ("b", False, {}),
        ("a", False, {}),
        ("aa", False, {}),
        ("bb", False, {}),
        ("aab", False, {}),
        ("abc", True, {"ou": ["abc"], "in": ["c"]}),
        ("abccccabcabcc", True, {"ou": ["abc"], "in": ["c"]}),
        ("abcabdabf", True, {"ou": ["abc", "abd", "abf"], "in": ["c", "d", "f"]}),
        (
            "abcabdabfffffabffabf",
            True,
            {"ou": ["abc", "abd", "abf"], "in": ["c", "d", "f"]},
        ),
        ("abcabdabffffccfdabcfabd", False, {}),
        ("sdfsd", False, {}),
    ],
    "*ф*и": [
        ("", False, {}),
        ("ffii", False, {}),
        ("ффии", True, {}),
        ("офщи", True, {}),
        ("wф0и", True, {}),
        ("pфи", False, {}),
    ],
    EMAIL_PATTERN: [
        (
            "john.smith@example.com",
            True,
            {"local": ["john.smith"], "domain": ["example.com"]},
        ),
        ("haha.com", False, {}),
    ],
    # GIANT_EMAIL_PATTERN: [
    #     (
    #         "john.smith@example.com",
    #         True,
    #         {"local": ["john.smith"], "domain": ["example.com"]},
    #     ),
    #     ("haha.com", False, {}),
    # ],
}


data_find_all = {
    "a*b": [
        ("a", []),
        ("ab", []),
        ("abo", []),
        ("abbaaabacb", [("abb", {}), ("aab", {}), ("acb", {})]),
        ("", []),
    ],
    "a...": [
        ("aaba", [("aa", {}), ("a", {})]),
        ("aaa", [("aaa", {})]),
        ("a", [("a", {})]),
        ("", []),
    ],
    "(<fi>[ab])[ab]...": [
        ("abca", [("ab", {"fi": ["a"]}), ("a", {"fi": ["a"]})]),
        (
            "bababa abba!abcd",
            [("bababa", {"fi": ["b"]}), ("abba", {"fi": ["a"]}), ("ab", {"fi": ["a"]})],
        ),
        ("b", [("b", {"fi": ["b"]})]),
        ("", []),
    ],
    "((<bs>b)|a)...": [
        ("abca", [("ab", {"bs": [None, "b"]}), ("a", {"bs": [None]})]),
        (
            "bababa abba!abcd",
            [
                ("bababa", {"bs": ["b", None, "b", None, "b", None]}),
                ("abba", {"bs": [None, "b", "b", None]}),
                ("ab", {"bs": [None, "b"]}),
            ],
        ),
        ("b", [("b", {"bs": ["b"]})]),
        ("", []),
    ],
    EMAIL_PATTERN: [
        (
            "john.smith@example.com",
            [
                (
                    "john.smith@example.com",
                    {"local": ["john.smith"], "domain": ["example.com"]},
                )
            ],
        ),
        ("haha.com", []),
        (
            "Hello! Here's my emial: my.cool.email.address@mail.ru! Feel free to reach me!",
            [
                (
                    "my.cool.email.address@mail.ru",
                    {"local": ["my.cool.email.address"], "domain": ["mail.ru"]},
                )
            ],
        ),
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
    # re = parse('b|((<gg>a)|%?%){2}?')
    pprint(re)
    tnfa = ast_to_tnfa(re)
    tnfa.dump_dot("tnfa.dot")
    tdfa = tnfa_to_tdfa(tnfa)
    tdfa.dump_dot("tdfa.dot")


def test_dfa3():
    # b|((<o>a)|\?)f*
    # re = parse("b|((<gg>a)|%?%){2}?f...")
    # re = parse("((<gg>a)|%?%|<gg>){2}")
    # re = parse("(<a>a)...(<b>a|c)c...")
    re = parse("(%+%?(d)...)?(((%(%|%[%|%{%)?d...(%)%|%]%|%}%)?)...)")
    # re = parse('b|((<gg>a)|%?%){2}?')
    # re = parse('(a|b|c){,}')
    # re = parse('()|((b|%?%|a%?%|a))')
    # re = parse("[abcd]e[ght]")
    # re = parse("[abcd]e")
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
    # pprint((re))
    tnfa = ast_to_tnfa(re)
    tnfa.dump_dot("tnfa.dot")
    tdfa = tnfa_to_tdfa(tnfa)
    tdfa.dump_dot("tdfa.dot")
    sim = tdfa.as_simulatable()
    print(sim)
    restored = sim.restore_regular_expression_via_k_path()
    print(restored)
    assert restored

    re = parse(restored)
    tnfa = ast_to_tnfa(re)
    tnfa.dump_dot("tnfa2.dot")
    tdfa = tnfa_to_tdfa(tnfa)
    tdfa.dump_dot("tdfa2.dot")
    sim = tdfa.as_simulatable()


# fmt: on


TEST_TNFA = False
DUMP_DOT = True


if __name__ == "__main__":
    # test_dfa0()
    # test_dfa1()
    # test_dfa2()
    # test_dfa3()
    test_regexes(data_full_match, data_find_all)
    print("DONE")
