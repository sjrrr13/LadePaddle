import json
import matplotlib.pyplot as plt

# with open('Output/greedy-7b-2.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         print(data['task_id'])
#         print("=" * 10)
#         print(data['pred_output'])
#         print()
#         print()

# W = [2, 3, 4, 5, 6, 7, 8]
# R = [1.81, 2.01, 2.14, 2.22, 2.26, 2.36, 2.42]
# R2 = [2.05, 2.20, 2.28, 2.33, 2.38, 2.42, 2.46]

# speed0 = [15.14, 15.14, 15.14, 15.14, 15.14, 15.14, 15.14]
# speed02 = [14.80, 14.80, 14.80, 14.80, 14.80, 14.80, 14.80]
# speed = [23.46, 23.92, 26.60, 27.13, 28.14, 29.94, 28.45]
# speed2 = [24.50, 25.58, 26.89, 26.69, 27.82, 27.61, 28.63]

# W = [3, 4, 5]
# R = [2.01, 2.14, 2.22]
# R1 = [2.08, 2.16, 2.24]
# R2 = [2.04, 2.14, 2.24]
# speed = [23.92, 26.60, 27.13]
# speed1 = [26.36, 26.25, 27.90]
# speed2 = [25.98, 26.02, 27.61]

W = [2, 3, 4]
speed_base = [23.46, 23.92, 26.60]
speed = [23.74, 25.82, 27.02]

r_base = [1.81, 2.01, 2.14]
r = [1.88, 2.03, 2.16]

plt.plot(W, speed_base, label="Baseline")
plt.plot(W, speed, label="Dynamic W")
# plt.plot(W, R2, label="100;+1")
plt.title("Speed for Dynamic W")
plt.xlabel("W")
plt.ylabel("tokens/s")
plt.legend()
plt.tight_layout()
plt.savefig("./Output/DW/Figs/7b-R.png")

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# ax1.plot(W, speed_base, label="Baseline")
# ax1.plot(W, speed, label="Dynamic W")
# ax1.legend()
# ax1.set_title('Speed')
# ax1.set_xlabel('W')
# ax1.set_ylabel('tokens/s')

# ax2.plot(W, r_base, label="Baseline")
# ax2.plot(W, r, label="Dynamic W")
# ax2.legend()
# ax2.set_title('R')
# ax2.set_xlabel('W')

# plt.suptitle("Results of Llama-2-7b")
# plt.subplots_adjust(wspace=0.5)
# plt.savefig("./Output/DW/Figs/7b.png")
