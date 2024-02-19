import re
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


SEPARATOR = "=" * 40


def improve_command(com: str) -> str:
    return com.replace("-v1", "-re").replace("-v2", "-smc").replace("-v3", "-lex")


def process(text: str):
    pattern = re.compile(r"(.+) -t input_(\d+).txt: (.+)")

    data = defaultdict(lambda: defaultdict(list))
    for trial in text.split(SEPARATOR):
        for command, size_s, time_s in pattern.findall(trial):
            command = improve_command(command)
            size = int(size_s)
            time = float(time_s)
            data[command][size].append(time)

    average = defaultdict(list)
    for command, values in data.items():
        for size, times in values.items():
            average[command].append((size, sum(times) / len(times)))
    return average


def print_data(average):
    for command, values in average.items():
        print(f"Command: {command}")
        for size, times in values:
            print(f"  Size: {size}, Times: {times}")
        print()


def visualize(ax, average, shuffled, log):
    log = bool(log)
    shuffled = bool(shuffled)

    for command, data_points in average.items():
        sizes, times = zip(*data_points)
        ax.plot(sizes, times, marker="o", label=command)

    ax.set_title(f"{shuffled=}, {log=}")
    ax.set_xscale("log" if log else "linear")
    ax.set_yscale("log" if log else "linear")

    ax.legend()


versions = [
    Path("timing_shuffled.txt").read_text(),
    Path("timing_not_shuffled.txt").read_text(),
]

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
fig.suptitle("Time (s) vs. Input File Lines")#, fontsize=20)
fig.supxlabel("Input File Lines")
fig.supylabel("Time")

for log in (0, 1):
    for shuffled in (0, 1):
        visualize(ax[shuffled][log], process(versions[shuffled]), shuffled, log)

plt.show()
