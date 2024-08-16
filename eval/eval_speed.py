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

SPEEDFILE = f"./Output/jsonls/{FILENAME}.jsonl"
SPEEDFIG = f"./Output/Figs/{FILENAME}_speed.png"

COMPRESSIONFILE = f"./Output/logs/{FILENAME}.log"
COMPRESSIONFIG = f"./Output/Figs/{FILENAME}_compression.png"

STR = "ratio: "

def draw(isSpeed, data, mean_d):
    plt.plot(data)
    plt.xlabel("Task")
    
    min_d = np.min(data)
    max_d = np.max(data)
    delta = (max_d - min_d) // 10 + 0.5
    plt.ylim(min_d-delta, max_d+delta)
    plt.axhline(y=mean_d, color='r', linestyle='--', label='Mean')
    plt.text(165, mean_d, str(mean_d), color='r', ha='left', va='bottom')
    
    if isSpeed:
        plt.ylabel("Speed")
        plt.title(f"Speed of {FILENAME}")
        plt.savefig(SPEEDFIG)
    else:
        plt.ylabel("Compression")
        plt.title(f"Compression of {FILENAME}")
        plt.savefig(COMPRESSIONFIG)


try:
    speeds = []

    with open(SPEEDFILE, "r") as f:
        for line in f:
            data = json.loads(line)
            speeds.append(data["speed"])
    
    speeds = np.array(speeds)
    speed = round(speeds.mean(), 2)
    print(f"Average speed: {speed:.2f} tokens/s")

    if DRAW:
        draw(True, speeds, speed)
except FileNotFoundError:
    print("No speed file found")

try:
    compressions = []
    
    with open(COMPRESSIONFILE, "r") as f:
        for line in f:
            if STR in line:
                line = line.rstrip()
                idx = line.find(STR) + len(STR)
                compressions.append(float(line[idx:]))

    compressions = np.array(compressions)
    compression = round(compressions.mean(), 2)
    print(f"Compression: {compression}")

    if DRAW:
        draw(False, compressions, compression)
except FileNotFoundError:
    print("No compression file found")
