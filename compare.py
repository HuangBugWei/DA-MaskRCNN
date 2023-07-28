import pickle
import torch, torchvision
from utils import postprocess

with open("og-result.pkl", "rb") as f:
    og = pickle.load(f)

with open("ts-result.pkl", "rb") as f:
    ts = pickle.load(f)

postprocess(ts, 1920/1333, 1.44, 1080, 1920)

for a, b in zip(og["instances"].pred_masks[:5], ts["pred_masks"][:5]):
    print(torch.equal(a, b))

print()

for a, b in zip(og["instances"].pred_boxes[:5], ts["pred_boxes"][:5]):
    print(torch.equal(a, b))