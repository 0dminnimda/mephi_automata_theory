import string
import random
import subprocess
from pathlib import Path


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CLEAR = "\033[0m"


MAIN = "main.py"


TEST_CASE = tuple[str, str, bool]


def test_args(prog: str, args: str, cases: list[TEST_CASE], verbose: int) -> bool:
    output = subprocess.run(
        f"{prog} {MAIN} {args}".split(" "),
        #shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        input="\n".join(it[0] for it in cases),
    )

    passed = True
    lines = output.stdout.splitlines()
    # print(*lines, sep="\n")
    i = 1
    for line, report, (prompt, name, should_match) in zip(
        lines, lines[len(cases) :], cases
    ):
        if (line == "OK") is should_match:
            if verbose:
                print(GREEN + f"  Test {i} passed: correctness" + CLEAR, repr(prompt))
        else:
            if verbose:
                print(RED + f"  Test {i} failed: correctness" + CLEAR, f"{line} != {should_match}", " - ", repr(prompt))
            passed = False

        if report == name:
            if verbose:
                print(GREEN + f"  Test {i} passed: report" + CLEAR, repr(prompt))
        else:
            if verbose:
                print(RED + f"  Test {i} failed: report" + CLEAR, f"{report!r} != {name!r}", repr(prompt))
            passed = False

        i += 1

    i -= 1
    if i != len(cases):
        if verbose:
            print(RED + f"  Only {i} out of {len(cases)} tests had output" + CLEAR)
        passed = False

    return passed


def test_all(prog_args: list[tuple[str, str]], cases: list[TEST_CASE], verbose: int) -> None:
    passed = True
    for prog, args in prog_args:
        print("Testing args:", prog, args)

        passed_this = test_args(prog, args, cases, verbose)

        if not passed_this:
            print(RED + "Args failed:" + CLEAR, prog, args)
        else:
            print(GREEN + "Args passed:" + CLEAR, prog, args)

        print()

        passed = passed and passed_this

    print("\n" + "=" * 80)
    if not passed:
        print(RED + "Some files failed!" + CLEAR)
    else:
        print(GREEN + "All files passed!" + CLEAR)


ALMOST_EVEERYTHING = string.digits + string.ascii_letters + string.punctuation + " "


def random_string(
    bounds: tuple[int, int] = (5, 25), alphabet: str = string.printable
) -> str:
    return "".join(random.choice(alphabet) for _ in range(random.randint(*bounds)))


NAME_FILLER = string.digits + string.ascii_letters


def random_name(bounds: tuple[int, int] = (5, 25)) -> str:
    return random_string((1, 1), string.ascii_letters) + random_string(
        (bounds[0] - 1, bounds[1]), NAME_FILLER
    )


def random_spaces(lower_bound: int = 1, upper_bound: int = 20) -> str:
    return " " * random.randint(lower_bound, upper_bound)


def generate_prompt(names_bounds, space_bounds, clazz, parents_bounds, parens, semi):
    specifiers = "", "private", "protected", "public"

    parents = [
        random_name(names_bounds) for _ in range(random.randint(*parents_bounds))
    ]
    duplicates = random.randint(0, 2 if parents else 0)
    for _ in range(duplicates):
        parents.append(random.choice(parents))

    pairs = ",".join(
        random_spaces(0, 5)
        + random.choice(specifiers)
        + random_spaces(*space_bounds)
        + parent
        + random_spaces(0, 5)
        for parent in parents
    )

    spaces1 = random_spaces(*space_bounds)
    spaces2 = random_spaces(*space_bounds)
    spaces3 = random_spaces(*space_bounds)
    spaces4 = random_spaces(*space_bounds)
    name = random_name(names_bounds)
    return (
        f"{clazz}{spaces1}{name}{spaces2}:{spaces3}{pairs}{spaces4}{parens}{semi}",
        name if duplicates else "",
    )


def generate_correct_prompt():
    return generate_prompt((5, 25), (1, 5), "class", (1, 10), "{}", ";")


def generate_semicorrect_prompt():
    classes = "clasS", "claSs", "clAss", "cLass", "Class", "cla", "c", "al", "ass", ""
    parens = "()", "[]", "{ }", "p", ""
    semi = ";;", " ;", ""
    return (
        generate_prompt(
            (0, 5),
            (0, 3),
            random.choice(classes),
            (0, 5),
            random.choice(parens),
            random.choice(semi),
        )[0],
        "",
    )


def generate_incorrect_prompt():
    return random_string((0, 100), ALMOST_EVEERYTHING), ""


def generate_prompts(n: int):
    result: list[TEST_CASE] = []

    for _ in range(n // 3):
        result.append((*generate_correct_prompt(), True))

    for _ in range(n // 3):
        result.append((*generate_semicorrect_prompt(), False))

    for _ in range(n // 3):
        result.append((*generate_incorrect_prompt(), False))

    return result


def generate_test_data(gen):
    gen_iter = iter(gen)
    for size in gen_iter:
        produce_input = Path(f"input_{size}.txt")
        if produce_input.exists():
            gen_iter.send(False)
            yield produce_input
            continue

        gen_iter.send(True)
        cases = next(gen_iter)
        with produce_input.open("w") as file:
            for i, it in enumerate(cases):
                if i:
                    file.write("\n")
                file.write(it[0])
            produce_input.write_text("\n".join(it[0] for it in cases))
        yield produce_input


def successive_test_case_generator(mx: int):
    yield 0
    do = yield
    if do:
        yield []

    yield len(USER_CASES)
    do = yield
    if do:
        yield USER_CASES

    for i in range(2, mx):
        size = 10**i
        yield (size//3)*3 + len(USER_CASES)
        do = yield
        if do:
            arr = generate_prompts(size) + USER_CASES
            random.shuffle(arr)
            yield arr


def time_runs(prog_args: list[tuple[str, str]], mx: int):
    for prog, args in prog_args:
        args += " -t"
        for file in generate_test_data(successive_test_case_generator(mx)):
            print(f"{prog} {args} {file}: ", end="", flush=True)
            subprocess.run(
                f"{prog} {MAIN} {args} {file}".split(" "),
                shell=True,
                encoding="utf-8",
            )


USER_CASES = [
    ("C", "", False),
    ("cL", "", False),
    ("clA", "", False),
    ("claS", "", False),
    ("clasS", "", False),
    ("class ", "", False),
    ("class a", "", False),
    ("class ab", "", False),
    ("class ab ", "", False),
    ("class ab  ", "", False),
    ("class ab  :", "", False),
    ("class ab  : ", "", False),
    ("class ab  :  ", "", False),
    ("class ab  :  g", "", False),
    ("class ab  :  gg", "", False),
    ("class ab  :  gg, ", "", False),
    ("class ab  :  gg, public", "", False),
    ("class ab  :  gg, public jj", "", False),
    ("class ab  :  gg, public jj {", "", False),
    ("class ab  :  gg, public jj {}", "", False),
    ("class ab  :  gg, public jj {};", "", True),
    ("class ab  :  gg, public jj, private jj {};", "ab", True),
    ("class ab  :  gg, public , private jj {};", "", True),  # hmm!
    ("class ab  :  gg, public , public, private jj {};", "ab", True),  # hmm!
    ("class ab  :  gg, public 0, private jj {};", "", False),
    ("class ab  :  gg, , private jj {};", "", False),
    ("class public: class, public, class {};", "public", True),
    ("", "", False),
]

PYTHON = "python"  # "python3.11"
PYPY = "pypy"

programs = [
    PYTHON,
    PYPY,
    PYPY,

    PYTHON,
    PYTHON,
]
args = [
    "-v1",
    "-v2",
    "-v3",

    "-v3",
    "-v2",
]
verbose = 0
do_timeing = False

prog_args = list(zip(programs, args))

if do_timeing:
    time_runs(prog_args, 7)
else:
    cases = generate_prompts(1_000) + USER_CASES
    test_all(prog_args, cases, verbose)
