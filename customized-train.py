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
from customizedTrainer import customAMPTrainer

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    train_target_loader = instantiate(cfg.dataloader.train_target)

    model = create_ddp_model(model, **cfg.train.ddp)
    # trainer = (customAMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, 
    #                                                                    train_loader, 
    #                                                                    train_target_loader, 
    #                                                                    optim)
    trainer = (customAMPTrainer)(model, 
                                train_loader, 
                                train_target_loader, 
                                optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),

            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,

            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    DatasetCatalog.register('steel_train', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/files-7000.txt", txt=True))
    DatasetCatalog.register('steel_train_target', lambda :  get_no_label_dicts("/home/aicenter/pytorch-CycleGAN-and-pix2pix/datasets/BIM2Real/trainB"))
    DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/B.txt", txt=True))
    # DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("./rebar-revit-auto-dataset/files-200.txt", txt=True))
    MetadataCatalog.get("steel_train").set(thing_classes=['intersection', 'spacing'])
    MetadataCatalog.get("steel_test").set(thing_classes=['intersection', 'spacing'])
    cfg.dataloader.train = LazyCall(build_detection_train_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names="steel_train"),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=True,
                                augmentations=[
                                    LazyCall(T.RandomBrightness)(intensity_min=0.4, intensity_max=1.6),
                                    LazyCall(T.RandomContrast)(intensity_min=0.4, intensity_max=1.6),
                                    LazyCall(T.RandomRotation)(angle=[-5, 5], expand=False, center=None, sample_style='range', interp=2),
                                    #LazyCall(T.ResizeScale)(min_scale=0.8, max_scale=1.2, target_height=1800, target_width=2400, interp=2),
                                    #LazyCall(T.RandomCrop)(crop_type="relative", crop_size=(720/1800, 1280/2400)),
                                    LazyCall(T.ResizeShortestEdge)(
                                        short_edge_length=(640, 672, 704, 736, 768, 800),
                                        sample_style="choice",
                                        max_size=1333,
                                    ),
                                    LazyCall(T.RandomFlip)(horizontal=True),
                                ],
                                image_format="BGR",
                                use_instance_mask=True,
                            ),
                            total_batch_size=10,
                            num_workers=4,
                            )
    cfg.dataloader.train_target = LazyCall(build_detection_train_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names="steel_train_target"),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=True,
                                augmentations=[
                                    LazyCall(T.RandomBrightness)(intensity_min=0.4, intensity_max=1.6),
                                    LazyCall(T.RandomContrast)(intensity_min=0.4, intensity_max=1.6),
                                    LazyCall(T.RandomRotation)(angle=[-5, 5], expand=False, center=None, sample_style='range', interp=2),
                                    #LazyCall(T.ResizeScale)(min_scale=0.8, max_scale=1.2, target_height=1800, target_width=2400, interp=2),
                                    #LazyCall(T.RandomCrop)(crop_type="relative", crop_size=(720/1800, 1280/2400)),
                                    LazyCall(T.ResizeShortestEdge)(
                                        short_edge_length=(640, 672, 704, 736, 768, 800),
                                        sample_style="choice",
                                        max_size=1333,
                                    ),
                                    LazyCall(T.RandomFlip)(horizontal=True),
                                ],
                                image_format="BGR",
                                use_instance_mask=True,
                            ),
                            total_batch_size=10,
                            num_workers=4,
                            )
    cfg.dataloader.train.total_batch_size = 8
    cfg.dataloader.train_target.total_batch_size = 8
    cfg.train.output_dir = "./DA-7000-3000-"
    cfg.train.max_iter = 100000
    cfg.train.checkpointer.period = 2000
    cfg.train.eval_period = 2000
    cfg.optimizer.lr = 0.00005
    # cfg.dataloader.test.dataset = LazyCall(get_detection_dataset_dicts)(names="steel_test", filter_empty=False)
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
    cfg.dataloader.evaluator = LazyCall(COCOEvaluator)(
                                    dataset_name="${..test.dataset.names}",
                                    output_dir=cfg.train.output_dir
                                )
    
    
    default_setup(cfg, args)
    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


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