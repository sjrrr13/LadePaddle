"""
Draw figures of speed and compression.
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    FILENAME = sys.argv[1]
else:
    print("Please input file name")
    exit()


DRAW = False
if len(sys.argv) > 2 and sys.argv[2] == "draw":
    DRAW = True

FILE = f"./Output/{FILENAME}.jsonl"
SPEEDFIG = f"./Output/Figs/{FILENAME}_speed.png"
COMPRESSIONFIG = f"./Output/Figs/{FILENAME}_compression.png"

TITLE_MAP = {
    "sample-7b": "Llama2-7b with LADE (Sampling)",
    "sample-13b": "Llama2-13b with LADE (Sampling)",
    "no-sample-7b": "Llama2-7b without LADE (Sampling)",
    "no-sample-13b": "Llama2-13b without LADE (Sampling)"
}


def draw(speeds, comps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
    title = TITLE_MAP[FILENAME]
    fig.suptitle(f"Speed and Compression of {title} on HumanEval")
    
    ax1.plot(speeds, 'tab:blue')
    ax1.set_xlabel("Task")
    ax1.set_ylabel("Speed (tokens/s)")

    ax2.plot(comps, 'tab:orange')
    ax2.set_xlabel("Task")
    ax2.set_ylabel("Compression")

    plt.tight_layout()
    plt.savefig(f"./Output/Figs/{FILENAME}.png")
    

try:
    speeds, comps = [], []
    windows = []

    with open(FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            speeds.append(data["speed"])
            comps.append(data["compression"])
            windows.append(tuple(data["windows"]))
    
    speeds = np.array(speeds)
    comps = np.array(comps)
    speed = round(speeds.mean(), 2)
    comp = round(comps.mean(), 2)
    print(f"Average speed: {speed:.2f} tokens/s\nAverage compression: {comp:.2f}")
    ave_window = []
    for w in windows:
        ave_window.append(sum(w) / len(w))
    ave_window = np.array(ave_window)
    print(f"Average: {ave_window.mean():.2f}; Var: {ave_window.var():4f}; Min: {ave_window.min():.4f}; Max: {ave_window.max():.4f}")

    if DRAW:
        draw(speeds, comps)
except FileNotFoundError:
    print("No speed file found")
