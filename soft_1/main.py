import sys
import time
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v", "--version", type=int, choices=[1, 2, 3], default=1, help="Set the version"
)
parser.add_argument("file", nargs="?", type=str, default="-", help="File to test")
parser.add_argument(
    "-o", "--output", nargs="?", type=str, default="-", help="Output file"
)
parser.add_argument("-t", "--time", action="store_true", help="Time the run and don't print any other")

args = parser.parse_args()


if args.file == "-":
    file = sys.stdin
elif Path(args.file).exists():
    file = Path(args.file).open("r")
else:
    print("File not found:", args.file)
    sys.exit(1)


if args.output == "-":
    output = sys.stdout
else:
    output = Path(args.output).open("w")


DIR = Path(__file__).parent


if args.version == 3:
    sys.path.insert(0, str(DIR / "lex"))
elif args.version == 2:
    sys.path.insert(0, str(DIR / "smc"))
else:
    sys.path.insert(0, str(DIR / "regex"))


from recognizer import match


if args.time:
    start = time.perf_counter()
    for line in file.read().splitlines():
        _ = match(line)
    print(time.perf_counter() - start)
else:
    results = []
    for line in file.read().splitlines():
        ok, res = match(line)
        if ok:
            print("OK", file=output)
        else:
            print("FAIL", file=output)

        if res is None:
            results.append("")
        else:
            results.append(res)

    print(*results, sep="\n", file=output)
