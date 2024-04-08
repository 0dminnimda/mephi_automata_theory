from parser import parse
from pprint import pprint
from tnfa import ast_to_tnfa


def test_one_regex(regex, cases):
    re = parse(regex)
    # print(re)
    tnfa = ast_to_tnfa(re)
    # pprint(asdict(tnfa), indent=4, width=200)
    for prompt, should_match in cases:
        assert (
            tnfa.run(prompt) is should_match
        ), f"{prompt!r} should {'not'if not should_match else ''} match {regex!r}"


def test_regexes(data):
    for regex, cases in data.items():
        test_one_regex(regex, cases)


data = {
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
    }
}


if __name__ == "__main__":
    test_regexes(data)
    print("SUCCESS!!")
