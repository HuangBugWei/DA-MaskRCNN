import os, random

# remove = os.listdir("/home/aicenter/maskrcnn/rebar-target-dataset/remove")
# target = os.listdir("/home/aicenter/maskrcnn/rebar-target-dataset/imgs")

# with open("/home/aicenter/maskrcnn/rebar-target-dataset/da-train-target.txt", "w") as f:
#     for item in target:
#         if item not in remove:
#             f.write(item + "\n")

tsne = []
with open("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/files-24826.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tsne.append(line.strip())

random.shuffle(tsne)

with open("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/feature-analysis.txt", "w") as f:
    for item in tsne[:500]:
        f.write(item + "\n")