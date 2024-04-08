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


# re = parse("b|((<gg>a)|%?%){2}?<gg>...")
re = parse("b|(a|%?%){2}?")
print(re)
tnfa = ast_to_tnfa(re)
# pprint(asdict(tnfa), indent=4, width=200)
assert tnfa.run("")
assert tnfa.run("b")
assert not tnfa.run("a")
assert not tnfa.run("?")
assert tnfa.run("aa")
assert tnfa.run("?a")
assert tnfa.run("a?")
assert tnfa.run("??")
assert not tnfa.run("??a")
assert not tnfa.run("?a?")
assert not tnfa.run("sdfsd")

re = parse("(a)")
print(re)
tnfa = ast_to_tnfa(re)
# pprint(asdict(tnfa), indent=4, width=200)
assert not tnfa.run("")
assert tnfa.run("a")
assert not tnfa.run("b")
assert not tnfa.run("ba")
assert not tnfa.run("ab")

re = parse("(abc)")
print(re)
tnfa = ast_to_tnfa(re)
# pprint(asdict(tnfa), indent=4, width=200)
assert not tnfa.run("")
assert not tnfa.run("a")
assert not tnfa.run("b")
assert not tnfa.run("c")
assert not tnfa.run("ab")
assert tnfa.run("abc")
assert not tnfa.run("abcd")

re = parse("(<gg>a)")
print(re)
tnfa = ast_to_tnfa(re)
# pprint(asdict(tnfa), indent=4, width=200)
assert not tnfa.run("")
assert tnfa.run("a")
assert not tnfa.run("b")
assert not tnfa.run("ba")
assert not tnfa.run("ab")

re = parse("(<gg>a){2}")
print(re)
tnfa = ast_to_tnfa(re)
# pprint(asdict(tnfa), indent=4, width=200)
assert not tnfa.run("")
assert not tnfa.run("a")
assert not tnfa.run("b")
assert not tnfa.run("ba")
assert not tnfa.run("ab")
assert tnfa.run("aa")
assert not tnfa.run("bb")
assert not tnfa.run("baa")
assert not tnfa.run("aba")
assert not tnfa.run("aaa")
assert not tnfa.run("bba")
assert not tnfa.run("bab")
assert not tnfa.run("abb")
assert not tnfa.run("aab")
assert not tnfa.run("bbb")

re = parse("b|((<gg>a)|%?%){2}?")
print(re)
tnfa = ast_to_tnfa(re)
# pprint(asdict(tnfa), indent=4, width=200)
assert tnfa.run("")
assert tnfa.run("b")
assert not tnfa.run("a")
assert not tnfa.run("?")
assert tnfa.run("aa")
assert tnfa.run("?a")
assert tnfa.run("a?")
assert tnfa.run("??")
assert not tnfa.run("??a")
assert not tnfa.run("?a?")
assert not tnfa.run("sdfsd")

print("SUCCESS!!")
