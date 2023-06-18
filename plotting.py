import os, json
import matplotlib.pyplot as plt

datapath = "./zinfandel/ablation-no-DA"

iterations = []
total_loss = []
lr = []
ap = []
with open(os.path.join(datapath, "test_source.json"), "r") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        # print(data)
        # if "total_loss" in data.keys():
        #     iterations.append(data["iteration"])
        #     total_loss.append(data["total_loss"])

        # if "loss_r3" in data.keys():
        #     iterations.append(data["iteration"])
        #     total_loss.append(data["loss_r3"])

        # if "lr" in data.keys():
        #     iterations.append(data["iteration"])
        #     lr.append(data["lr"])

        if "segm" in data.keys():
            # ap.append(data["segm/AP50"])
            # iterations.append(data["iteration"])
            ap.append(data["segm"]["AP50"])
            iterations.append((idx // 4 + 1) * 2500 - 1)
            

# fig, ax = plt.subplots()
# ax.plot(iterations, total_loss)
# fig.savefig(os.path.join(datapath, "loss_metrics.png"))

# fig, ax = plt.subplots()
# ax.plot(iterations, lr)
# fig.savefig(os.path.join(datapath, "lr_metrics.png"))

fig, ax = plt.subplots()
ax.plot(iterations, ap)
fig.savefig(os.path.join(datapath, "ap_source_metrics.png"))