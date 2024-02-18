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


def test_args(args: str, cases: list[TEST_CASE], verbose: int) -> bool:
    output = subprocess.run(
        f"python {MAIN} {args}".split(" "),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        input="\n".join(it[0] for it in cases),
    )

    passed = True
    lines = output.stdout.splitlines()
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

    return passed


def test_all(args: list[str], cases: list[TEST_CASE], verbose: int) -> None:
    passed = True
    for arg in args:
        print("Testing args:", arg)

        passed_this = test_args(arg, cases, verbose)

        if not passed_this:
            print(RED + "Args failed:" + CLEAR, arg)
        else:
            print(GREEN + "Args passed:" + CLEAR, arg)

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


def generate_prompt(names_bounds, clazz, parens, semi):
    specifiers = "", "private", "protected", "public"

    parents = [random_name(names_bounds) for _ in range(random.randint(1, 10))]
    duplicates = random.randint(0, 2)
    for _ in range(duplicates):
        parents.append(random.choice(parents))

    pairs = ",".join(
        random_spaces(0, 5)
        + f"{random.choice(specifiers)} {parent}"
        + random_spaces(0, 5)
        for parent in parents
    )

    spaces1 = random_spaces(1, 5)
    spaces2 = random_spaces(1, 5)
    spaces3 = random_spaces(1, 5)
    spaces4 = random_spaces(1, 5)
    name = random_name(names_bounds)
    return (
        f"{clazz}{spaces1}{name}{spaces2}:{spaces3}{pairs}{spaces4}{parens}{semi}",
        name if duplicates else "",
    )


def generate_correct_prompt():
    return generate_prompt((5, 25), "class", "{}", ";")


def generate_semicorrect_prompt():
    classes = "clasS", "claSs", "clAss", "cLass", "Class", "cla", "c", "al", "ass", ""
    parens = "()", "[]", "{ }", "p", ""
    semi = ";;", " ;", ""
    return generate_prompt(
        (0, 5), random.choice(classes), random.choice(parens), random.choice(semi)
    )[0], ""


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


args = [
    "-v1",
    "-v2",
]
cases = generate_prompts(1_000_000)
cases += [
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
    ("class ab  :  gg, public 0, private jj {};", "", False),
    ("class ab  :  gg, , private jj {};", "", False),
]
verbose = 0
produce_input: Path | None = Path("input.txt")

if produce_input is not None:
    with produce_input.open("w") as file:
        for i, it in enumerate(cases):
            if i:
                file.write("\n")
            file.write(it[0])
        produce_input.write_text("\n".join(it[0] for it in cases))
else:
    test_all(args, cases, verbose)
