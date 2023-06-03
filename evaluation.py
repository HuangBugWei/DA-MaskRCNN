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

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            # model, instantiate(cfg.dataloader.test), None
        )
        # with open(os.path.join(cfg.train.output_dir, "test_target.json"), "a") as f:
        #     json.dump(ret, f)
        #     f.write("\n")
        print_csv_format(ret)
        return ret

def do_source_test(cfg, model):
    if "evaluator_source" in cfg.dataloader and "test_source" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test_source), instantiate(cfg.dataloader.evaluator_source)
        )
        # with open(os.path.join(cfg.train.output_dir, "test_source.json"), "a") as f:
        #     json.dump(ret, f)
        #     f.write("\n")
        print_csv_format(ret)
        return ret

def do_validation_loss(cfg, model):
    if "test_source" in cfg.dataloader:
        losses = customLossEval(model, instantiate(cfg.dataloader.test_source), domainSource=True)
        with open(os.path.join(cfg.train.output_dir, "test_source.json"), "a") as f:
            json.dump(losses, f)
            f.write("\n")
    # if "test" in cfg.dataloader:
    #     losses = customLossEval(model, instantiate(cfg.dataloader.test), domainSource=False)
    #     with open(os.path.join(cfg.train.output_dir, "test_target.json"), "a") as f:
    #         json.dump(losses, f)
    #         f.write("\n")


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)


    DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-dataset", txt=False))
    DatasetCatalog.register('steel_test_source', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/rebar-revit-auto-test.txt", txt=True))
    # DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("./rebar-revit-auto-dataset/files-200.txt", txt=True))
    
    MetadataCatalog.get("steel_test").set(thing_classes=['intersection', 'spacing'])
    MetadataCatalog.get("steel_test_source").set(thing_classes=['intersection', 'spacing'])
    
    
    cfg.model.roi_heads.num_classes = 2
    

    cfg.dataloader.test = LazyCall(build_detection_test_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names="steel_test", filter_empty=False),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=False,
                                augmentations=[
                                    LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                                ],
                                image_format="${...train.mapper.image_format}",
                            ),
                            num_workers=4,
                            )
    cfg.dataloader.test_source = LazyCall(build_detection_test_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names="steel_test_source", filter_empty=False),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=True,
                                # use_instance_mask=True,
                                augmentations=[
                                    LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                                ],
                                image_format="${...train.mapper.image_format}",
                            ),
                            num_workers=4,
                            )
    cfg.train.output_dir = "./ablation-no-DA"
    cfg.dataloader.evaluator = LazyCall(COCOEvaluator)(
                                    dataset_name="${..test.dataset.names}",
                                    output_dir=os.path.join(cfg.train.output_dir,
                                                            "test_target")
                                )
    cfg.dataloader.evaluator_source = LazyCall(COCOEvaluator)(
                                    dataset_name="${..test_source.dataset.names}",
                                    output_dir=os.path.join(cfg.train.output_dir,
                                                            "test_source")
                                )
    
    cfg.train.init_checkpoint = os.path.join(cfg.train.output_dir,
                                             "model_final.pth")
    
    default_setup(cfg, args)
    
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    print(do_test(cfg, model))
    print(do_source_test(cfg, model))
    


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