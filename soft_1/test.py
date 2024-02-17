import string
import random
import subprocess
from pathlib import Path


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CLEAR = "\033[0m"


TEST_CASE = tuple[str, bool]


def test_one(file: Path, prompt: str, expected: bool) -> bool:
    output = subprocess.run(
        f"python {file}".split(" "),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        input=prompt,
    )

    if bool(output.returncode) is expected:
        print(RED + "  Test failed:" + CLEAR, repr(prompt))
        print(output.stdout)
        return False

    print(GREEN + "  Test passed:" + CLEAR, repr(prompt))
    return True


def test_file(file: Path, cases: list[TEST_CASE]) -> bool:
    assert file.suffix == ".py"

    passed = all(test_one(file, *case) for case in cases)

    if not passed:
        print(RED + "File failed:" + CLEAR, file)
    else:
        print(GREEN + "File passed:" + CLEAR, file)

    return passed


def test_all(files: list[Path], cases: list[TEST_CASE]) -> None:
    passed = all(test_file(file, cases) for file in files)

    print("\n" + "=" * 80)
    if not passed:
        print(RED + "Some files failed!" + CLEAR)
    else:
        print(GREEN + "All files passed!" + CLEAR)


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


def generate_correct_prompt() -> str:
    specifiers = "", "private", "protected", "public"

    pairs = ",".join(
        random_spaces(0, 5)
        + f"{random.choice(specifiers)} {random_name()}"
        + random_spaces(0, 5)
        for _ in range(random.randint(1, 10))
    )

    spaces1 = random_spaces(1, 5)
    spaces2 = random_spaces(1, 5)
    spaces3 = random_spaces(1, 5)
    spaces4 = random_spaces(1, 5)
    name = random_name()
    return f"class{spaces1}{name}{spaces2}:{spaces3}{pairs}{spaces4}{{}};"


def generate_incorrect_prompt() -> str:
    return random_string((0, 100))


def generate_prompts(n: int) -> list[TEST_CASE]:
    result = []

    for _ in range(n // 2):
        result.append((generate_correct_prompt(), True))

    for _ in range(n // 2):
        result.append((generate_incorrect_prompt(), False))

    return result


files = [Path("as_regex.py")]
prompts = [] + generate_prompts(100)

test_all(files, prompts)
