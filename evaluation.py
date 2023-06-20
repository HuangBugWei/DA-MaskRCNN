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
import torch, argparse, copy, time
from customizedEvalHook import customLossEval, customEvalHook

logger = logging.getLogger("detectron2")

def do_testing(cfg, model, iter, dataloader, evaluator, dataset_name):
    lastIter = 0
    try:
        with open(os.path.join(cfg.train.output_dir, f"{dataset_name}-ap.json"), "r") as f:
            lines = f.readlines()
            lastLog = json.loads(lines[-1].strip())
            lastIter = int(lastLog["iter"])
    except:
        print("target file not found")

    if lastIter >= int(iter):
        print("skipping since early log has already been recorded")
        return None
    ret = inference_on_dataset(model, dataloader, evaluator)

    ret["iter"] = iter
    with open(os.path.join(cfg.train.output_dir, f"{dataset_name}-ap.json"), "a") as f:
        json.dump(ret, f)
        f.write("\n")
    print_csv_format(ret)
    return ret

def build_loader_and_evaluator(cfg, dataset_name):
    return instantiate(LazyCall(build_detection_test_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names=dataset_name, filter_empty=False),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=False,
                                augmentations=[
                                    LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                                ],
                                image_format="BGR",
                            ),
                            num_workers=4,
                            )), instantiate(LazyCall(COCOEvaluator)(
                                    dataset_name=dataset_name,
                                    output_dir=os.path.join(cfg.train.output_dir),
                                    allow_cached_coco=False,
                                ))

def main(args):
    cfg = LazyConfig.load(args.config_file)
    # cfg = LazyConfig.apply_overrides(cfg, args.opts)

    DatasetCatalog.clear()
    MetadataCatalog.clear()
    # DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/test.txt", txt = True))
    # DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset", txt = False))
    if args.train:
        DatasetCatalog.register('steel_train', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/target-train.txt", txt = True))
        MetadataCatalog.get("steel_train").set(thing_classes=['intersection', 'spacing'])
        trainloader, traineval = build_loader_and_evaluator(cfg, "steel_train")
    
    DatasetCatalog.register('steel_val', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/target-val.txt", txt = True))
    MetadataCatalog.get("steel_val").set(thing_classes=['intersection', 'spacing'])
    valloader, valeval = build_loader_and_evaluator(cfg, "steel_val")

    DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/target-slab.txt", txt = True))
    MetadataCatalog.get("steel_test").set(thing_classes=['intersection', 'spacing'])
    testloader, testval = build_loader_and_evaluator(cfg, "steel_test")
    
    if args.bim:
        DatasetCatalog.register('steel_test_bim', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/rebar-revit-auto-test.txt", txt = True))
        MetadataCatalog.get("steel_test_bim").set(thing_classes=['intersection', 'spacing'])
        bimloader, bimeval = build_loader_and_evaluator(cfg, "steel_test_bim")
    
        
    cfg.model.roi_heads.num_classes = 2
    cfg.train.output_dir = args.input
    cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    # model.to("cpu")

    # default_setup(cfg, args)
    
    for ckpt in sorted(os.listdir(cfg.train.output_dir)):
        if ckpt.startswith("model") and ckpt.endswith(".pth"):
            print(f"now loading {ckpt}")
            iter = os.path.splitext(os.path.basename(ckpt))[0].split("_")[1]
            if iter == "final":
                break
            print(f"now iter {iter}")
            cfg.train.init_checkpoint = os.path.join(cfg.train.output_dir, ckpt)
            DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
            if args.train:
                print("train-ap")
                print(do_testing(cfg, model, iter, trainloader, traineval, "steel_train"))
            print("val-ap")
            print(do_testing(cfg, model, iter, copy.deepcopy(valloader), copy.deepcopy(valeval), "steel_val"))
            print("test-ap")
            print(do_testing(cfg, model, iter, copy.deepcopy(testloader), copy.deepcopy(testval), "steel_test"))
            if args.bim:
                print("bim-ap")
                print(do_testing(cfg, model, iter, copy.deepcopy(bimloader), copy.deepcopy(bimeval), "steel_test_bim"))
    
def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type = str, required = True)
    parser.add_argument("-t", "--train", action='store_true')
    parser.add_argument("-b", "--bim", action='store_true')
    parser.add_argument("--num-gpus", type = int, default = 1)
    parser.add_argument("--config-file", required = True, metavar="FILE", help="path to config file")

    return parser.parse_args()

if __name__ == "__main__":
    # args = default_argument_parser().parse_args()
    args = parser()
    start = time.time()
    launch(
        main,
        args.num_gpus,
        # num_machines=args.num_machines,
        # machine_rank=args.machine_rank,
        # dist_url=args.dist_url,
        args=(args,),
    )
    print(f"total time {time.time() - start} s")