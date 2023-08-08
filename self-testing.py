import os, json
import matplotlib.pyplot as plt
from plotting import *
import matplotlib

fig, ax = plt.subplots(figsize=(8,6))

datapath = "/home/aicenter/DA-MaskRCNN/0626-gemini/ablation-DA-25k"
iterations = []
ap = []
with open(os.path.join(datapath, "steel_val-0.5ap.json"), "r") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        try:
            i = int(data["iter"])
            if i not in iterations:
                iterations.append(int(data["iter"]))
                ap.append(data["segm"]["AP50"])
        except:
            pass
ax.plot(iterations, ap, marker="o", label="25k", color = "royalblue")

datapath = "/home/aicenter/DA-MaskRCNN/ablation-DA-25k-5k"
iterations = []
ap = []
with open(os.path.join(datapath, "steel_val-0.5ap.json"), "r") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        try:
            i = int(data["iter"])
            if i not in iterations:
                iterations.append(int(data["iter"]))
                ap.append(data["segm"]["AP50"])
        except:
            pass
ax.plot(iterations, ap, marker="o", label="5k", color = "darkorange")

datapath = "/home/aicenter/DA-MaskRCNN/ablation-DA-25k-1k"
iterations = []
ap = []
with open(os.path.join(datapath, "steel_val-0.5ap.json"), "r") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        try:
            i = int(data["iter"])
            if i not in iterations:
                iterations.append(int(data["iter"]))
                ap.append(data["segm"]["AP50"])
        except:
            pass
ax.plot(iterations, ap, marker="o", label="1k", color = "red")

datapath = "/home/aicenter/DA-MaskRCNN/ablation-DA-25k-500"
iterations = []
ap = []
with open(os.path.join(datapath, "steel_val-0.5ap.json"), "r") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        try:
            i = int(data["iter"])
            if i not in iterations:
                iterations.append(int(data["iter"]))
                ap.append(data["segm"]["AP50"])
        except:
            pass
ax.plot(iterations, ap, marker="o", label="500", color = "limegreen")


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14


# ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.89), fontsize = MEDIUM_SIZE)
ax.legend(fontsize = MEDIUM_SIZE)
ax.set_xlim(left=0, right=60000)
ax.set_xlabel("iterations", fontsize = BIGGER_SIZE)
ax.set_ylabel("AP50 (%)", fontsize = BIGGER_SIZE)
plt.xticks(fontsize = MEDIUM_SIZE)
plt.yticks(fontsize = MEDIUM_SIZE)
# ax.grid()
datapath = "/home/aicenter/DA-MaskRCNN/0626-gemini"
fig.savefig(os.path.join(datapath, "new-bim-data-num-compare-val.png"))