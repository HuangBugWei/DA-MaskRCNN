import os, time, cv2, torch, argparse
import numpy as np
from torch import Tensor
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate, LazyCall
from detectron2.structures import Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.transforms as T
from detectron2.export import dump_torchscript_IR, scripting_with_instances
from detectron2.export.torchscript_patch import freeze_training_mode, patch_instances

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type = str, required = True)
    parser.add_argument("-m", "--model", type = str, required = True)
    parser.add_argument("-o", "--output", type = str, required = True)
    parser.add_argument("--num-gpus", type = int, default = 1)
    parser.add_argument("--thres", type = float, default = 0.5)
    parser.add_argument("--config-file", required = True, metavar="FILE", help="path to config file")

    return parser.parse_args()


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg.model.roi_heads.box_predictor.test_score_thresh = args.thres
    # cfg.model.backbone.bottom_up._out_features = \
    #     list(cfg.model.backbone.bottom_up._out_features)
    if args.num_gpus > 1:
        cfg.model.backbone.bottom_up.stem.norm = \
        cfg.model.backbone.bottom_up.stages.norm = "SyncBN"
        cfg.model.backbone.norm = "SyncBN"
    else:
        cfg.model.backbone.bottom_up.stem.norm = \
        cfg.model.backbone.bottom_up.stages.norm = "BN"
        cfg.model.backbone.norm = "BN"

    model = instantiate(cfg.model)
    
    model.backbone.bottom_up._out_features = \
        list(model.backbone.bottom_up._out_features)
    model.proposal_generator.in_features = \
        list(model.proposal_generator.in_features)
    model.proposal_generator.anchor_generator.strides = \
        list(model.proposal_generator.anchor_generator.strides)
    model.proposal_generator.anchor_generator.strides = \
        list(model.proposal_generator.anchor_generator.strides)
    model.roi_heads.in_features = model.roi_heads.box_in_features = \
        list(model.roi_heads.box_in_features)
    model.roi_heads.mask_in_features = \
        list(model.roi_heads.mask_in_features)
    
    # model.to(cfg.train.device)
    model.to("cpu")

    # cfg.train.init_checkpoint = args.model
    DetectionCheckpointer(model).load(args.model)
    augs = instantiate(LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333))
    model.eval()
    fields = {
            "proposal_boxes": Boxes,
            "objectness_logits": Tensor,
            "pred_boxes": Boxes,
            "scores": Tensor,
            "pred_classes": Tensor,
            "pred_masks": Tensor,
        }
    # with freeze_training_mode(model), patch_instances(fields):
    #     mymodel = torch.jit.script(model)
    im = cv2.imread("/home/aicenter/DA-MaskRCNN/online-steel-image-crop/imgs/34.jpg")
    # mymodel = torch.jit.trace(model, (torch.rand(1)))
    # mymodel = torch.jit.script(model)
    mymodel = scripting_with_instances(model, fields)
    # mymodel = dump_torchscript_IR(model, "/home/aicenter/DA-MaskRCNN/torchscript")
    print(mymodel)
    mymodel.save("/home/aicenter/DA-MaskRCNN/torchscript/da-scripting-model.pt")




if __name__ == "__main__":
    args = parser()
    main(args)