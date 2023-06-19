#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.
This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.
Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging, os, json, glob
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate, LazyCall
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.transforms as T
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator

from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from utils import get_rebar_dicts, get_no_label_dicts
from customizedTrainer import customAMPTrainer, customSimpleTrainer
import torch
from customizedEvalHook import customLossEval, customEvalHook

logger = logging.getLogger("detectron2")

def do_testing(model, iter, dataloader, evaluator):
    lastIter = 0
    try:
        with open(os.path.join(evaluator.output_dir, f"{evaluator.dataset_name}-ap.json"), "r") as f:
            lines = f.readlines()
            lastLog = json.loads(lines[-1].strip())
            lastIter = int(lastLog["iter"])
    except:
        print("target file not found")

    if lastIter >= int(iter):
        print("skipping since early log has already been recorded")
        return None
    ret = inference_on_dataset(
        model, instantiate(dataloader), instantiate(evaluator)
    )
    ret["iter"] = iter
    with open(os.path.join(evaluator.output_dir, f"{evaluator.dataset_name}-ap.json"), "a") as f:
        json.dump(ret, f)
        f.write("\n")
    print_csv_format(ret)
    return ret

def build_loader_and_evaluator(cfg, dataset_name):
    return LazyCall(build_detection_test_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names=dataset_name, filter_empty=False),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=False,
                                augmentations=[
                                    LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                                ],
                                image_format="BGR",
                            ),
                            num_workers=4,
                            ), LazyCall(COCOEvaluator)(
                                    dataset_name=dataset_name,
                                    output_dir=os.path.join(cfg.train.output_dir),
                                    allow_cached_coco=False,
                                )

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    DatasetCatalog.clear()
    MetadataCatalog.clear()
    # DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/test.txt", txt = True))
    # DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset", txt = False))
    DatasetCatalog.register('steel_train', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/target-train.txt", txt = True))
    DatasetCatalog.register('steel_val', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/target-val.txt", txt = True))
    DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/target-slab.txt", txt = True))
    DatasetCatalog.register('steel_test_bim', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/rebar-revit-auto-test.txt", txt = True))
    
    MetadataCatalog.get("steel_train").set(thing_classes=['intersection', 'spacing'])
    MetadataCatalog.get("steel_val").set(thing_classes=['intersection', 'spacing'])
    MetadataCatalog.get("steel_test").set(thing_classes=['intersection', 'spacing'])
    MetadataCatalog.get("steel_test_bim").set(thing_classes=['intersection', 'spacing'])
        
    cfg.model.roi_heads.num_classes = 2
    cfg.train.output_dir = "/home/aicenter/DA-MaskRCNN/0619-zinfandel/ablation-DA-25k-061815"
    cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5

    trainloader, traineval = build_loader_and_evaluator(cfg, "steel_train")
    valloader, valeval = build_loader_and_evaluator(cfg, "steel_val")
    testloader, testval = build_loader_and_evaluator(cfg, "steel_test")
    bimloader, bimeval = build_loader_and_evaluator(cfg, "steel_test_bim")
    
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    # model.to("cpu")

    # default_setup(cfg, args)
    print(os.listdir(cfg.train.output_dir))
    for ckpt in sorted(os.listdir(cfg.train.output_dir)):
        if ckpt.startswith("model") and ckpt.endswith(".pth"):
            print(f"now loading {ckpt}")
            iter = os.path.splitext(os.path.basename(ckpt))[0].split("_")[1]
            if iter == "final":
                break
            print(f"now iter {iter}")
            cfg.train.init_checkpoint = os.path.join(cfg.train.output_dir, ckpt)
            DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
            print("train-ap")
            print(do_testing(model, iter, trainloader, traineval))
            print("val-ap")
            print(do_testing(model, iter, valloader, valeval))
            print("test-ap")
            print(do_testing(model, iter, testloader, testval))
            print("bim-ap")
            print(do_testing(model, iter, bimloader, bimeval))
    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )