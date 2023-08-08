import os, json
import matplotlib.pyplot as plt

def plotLoss(ax, datapath, label=None):
    iterations = []
    total_loss = []
    with open(os.path.join(datapath, "metrics.json"), "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            # print(data)
            if "total_loss" in data.keys():
                iterations.append(data["iteration"])
                if "loss_r3" in data.keys():
                    total_loss.append(data["total_loss"]
                                    - data["loss_r3"]
                                    - data["loss_r4"]
                                    - data["loss_r5"])
                else:
                    total_loss.append(data["total_loss"])
    if label is None:
        ax.plot(iterations, total_loss)
    else:
        ax.plot(iterations, total_loss, label=label)
    

def plotAP(ax, aptype, datapath, label=None):
    iterations = []
    ap = []
    with open(os.path.join(datapath, f"{aptype}-0.5ap.json"), "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            try:
                i = int(data["iter"])
                if i not in iterations:
                    iterations.append(int(data["iter"]))
                    ap.append(data["segm"]["AP50"])
            except:
                pass
    
    if label is None:
        ax.plot(iterations, ap, marker = "o", color="darkorange")
    else:
        ax.plot(iterations, ap, marker = "o", label=label, color="darkorange")

def plotAddAP(ax, datapath, label=None):
    iterations = []
    ap = []
    with open(os.path.join(datapath, "steel_test-ap.json"), "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            try:
                i = int(data["iter"])
                if i not in iterations:
                    iterations.append(int(data["iter"]))
                    ap.append(data["segm"]["AP50"])
            except:
                pass
    with open(os.path.join(datapath, "steel_val-ap.json"), "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            try:
                i = int(data["iter"])
                if i in iterations:
                    
                    ap[idx] += (data["segm"]["AP50"])
            except:
                pass
    print(iterations[ap.index(max(ap))])
    if label is None:
        ax.plot(iterations, ap)
    else:
        ax.plot(iterations, ap, label=label)

def plotLR(ax, datapath, label=None):
    iterations = []
    lr = []
    with open(os.path.join(datapath, "metrics.json"), "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            if "lr" in data.keys():
                iterations.append(data["iteration"])
                lr.append(data["lr"])

    if label is None:
        ax.plot(iterations, lr)
    else:
        ax.plot(iterations, lr, label=label)

fig, ax = plt.subplots(figsize=(8, 5))

# color = 'tab:red'
# ax.set_xlabel('iterations')
# ax.set_ylabel('loss')
# datapath = "/home/aicenter/DA-MaskRCNN/0626-gemini/ablation-vanilla-235"
# iterations = []
# total_loss = []
# with open(os.path.join(datapath, "metrics.json"), "r") as f:
#     for idx, line in enumerate(f):
#         data = json.loads(line)
#         # print(data)
#         if "total_loss" in data.keys():
#             iterations.append(data["iteration"])
#             if "loss_r3" in data.keys():
#                 total_loss.append(data["total_loss"]
#                                 - data["loss_r3"]
#                                 - data["loss_r4"]
#                                 - data["loss_r5"])
#             else:
#                 total_loss.append(data["total_loss"])
# ax.plot(iterations, total_loss, color=color, label="training (loss)")
# ax.tick_params(axis='y')

# ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('AP50')  # we already handled the x-label with ax1
# iterations = []
# ap = []
# with open(os.path.join(datapath, "steel_val-0.5ap.json"), "r") as f:
#     for idx, line in enumerate(f):
#         data = json.loads(line)
#         try:
#             i = int(data["iter"])
#             if i not in iterations:
#                 iterations.append(int(data["iter"]))
#                 ap.append(data["segm"]["AP50"])
#         except:
#             pass
# ax2.plot(iterations, ap, color=color, label="validation (AP50)", marker='o')
# ax2.tick_params(axis='y')

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# ax.grid()
# ax.set_xlim(left=0, right=60000)
# fig.legend(loc='upper right', bbox_to_anchor=(0.93, 0.9))
# # fig.legend(loc='upper right')
# fig.savefig(os.path.join(datapath, "learning-curve.png"))

# datapath = "/home/aicenter/DA-MaskRCNN/0626-gemini/ablation-no-DA-25k"
# plotLoss(ax, datapath, label="training")
# iterations = []
# total_loss = []
# with open(os.path.join(datapath, "bim-val-loss.json"), "r") as f:
#     for idx, line in enumerate(f):
#         data = json.loads(line)
#         # print(data)
#         if "total_loss" in data.keys():
#             iterations.append(int(data["iter"]))
#             if "loss_r3" in data.keys():
#                 total_loss.append(data["total_loss"]
#                                 - data["loss_r3"]
#                                 - data["loss_r4"]
#                                 - data["loss_r5"])
#             else:
#                 total_loss.append(data["total_loss"])
# ax.plot(iterations, total_loss, label="validation", marker='o')
# ax.set_xlabel("iterations")
# ax.set_ylabel("total loss")
# ax.set_title("learning curve")
# ax.legend()
# ax.set_xlim(left=0, right=60000)
# ax.grid()
# fig.savefig(os.path.join(datapath, "learning-curve2.png"))

# datapath = "/home/aicenter/DA-MaskRCNN/ablation-vanilla-235"
# plotLoss(ax, datapath, label="original")
# datapath = "/home/aicenter/DA-MaskRCNN/ablation-no-DA-25k"
# plotLoss(ax, datapath, label="BIM")
# datapath = "/home/aicenter/DA-MaskRCNN/ablation-DA-25k"
# plotLoss(ax, datapath, label="BIM w/ DA")
# ax.set_xlabel("iterations")
# ax.set_ylabel("total loss")
# ax.set_title("training curve")
# ax.legend()
# ax.grid()
# ax.set_xlim(left=0, right=60000)
# fig.savefig(os.path.join("/home/aicenter/DA-MaskRCNN/ablation-result", "cross_loss_metrics.png"))


# datapath = "/home/aicenter/DA-MaskRCNN/ablation-transfer-5"
# plotAP(ax, "steel_val", datapath, "validation")
# plotAP(ax, "steel_test", datapath, "test")

# datapath = "/home/aicenter/DA-MaskRCNN/0626-gemini/ablation-vanilla-235"
# plotAP(ax, "steel_val", datapath, "original")
# datapath = "/home/aicenter/DA-MaskRCNN/0626-gemini/ablation-no-DA-25k"
# plotAP(ax, "steel_test_bim", datapath, "BIM")
# datapath = "/home/aicenter/DA-MaskRCNN/0626-gemini/ablation-DA-25k"
# plotAP(ax, "steel_test_bim", datapath, "BIM w/ DA")
# datapath = "/home/aicenter/DA-MaskRCNN/ablation-transfer-4"
# plotAP(ax, "steel_test", datapath, "transfer-4")
# datapath = "/home/aicenter/DA-MaskRCNN/ablation-transfer-5"
# plotAP(ax, "steel_test", datapath, "transfer-5")

# ax.set_xlabel("iterations")
# ax.set_ylabel("AP50")
# ax.set_title("AP performance on test data set")
# # ax.legend()
# ax.set_xlim(0, 60000)
# ax.grid()
# # fig.savefig(os.path.join("/home/aicenter/DA-MaskRCNN/ablation-result", "cross-test-ap.png"))
# fig.savefig(os.path.join(datapath, "real-val-ap.png"))

# fig, ax = plt.subplots()
# datapath = "/home/aicenter/DA-MaskRCNN/ablation-vanilla-235"
# plotAddAP(ax, datapath, "original")
# datapath = "/home/aicenter/DA-MaskRCNN/ablation-no-DA-25k"
# plotAddAP(ax, datapath, "BIM")
# datapath = "/home/aicenter/DA-MaskRCNN/ablation-DA-25k"
# plotAddAP(ax, datapath, "BIM w/ DA")

# ax.set_xlabel("iterations")
# ax.set_ylabel("AP50")
# ax.set_title("overall data set performance")
# ax.legend()
# fig.savefig(os.path.join(datapath, "cross_overall_ap.png"))

# datapath = "/home/aicenter/DA-MaskRCNN/ablation-vanilla-235"
# plotLR(ax, datapath, label="original")
# datapath = "/home/aicenter/DA-MaskRCNN/ablation-no-DA-25k"
# plotLoss(ax, datapath, label="BIM")
# datapath = "/home/aicenter/DA-MaskRCNN/ablation-DA-25k"
# plotLoss(ax, datapath, label="BIM w/ DA")
# ax.set_xlabel("iterations")
# ax.set_ylabel("learning rate")
# ax.set_title("learning rate schedule")
# ax.legend()
# ax.grid()
# ax.set_xlim(left=0, right=60000)
# fig.savefig(os.path.join("/home/aicenter/DA-MaskRCNN/ablation-result", "lr-schedule.png"))



