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

# from detectron2.data import MetadataCatalog, DatasetCatalog
# import detectron2.data.transforms as T
# from detectron2.structures import BoxMode
# from detectron2.evaluation import COCOEvaluator

# from detectron2.data import (
#     DatasetMapper,
#     build_detection_test_loader,
#     build_detection_train_loader,
#     get_detection_dataset_dicts,
# )

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import WarmupParamScheduler

from utils import get_rebar_dicts, get_no_label_dicts
from customizedComponents.customizedTrainer import customAMPTrainer, customSimpleTrainer
# import torch
from customizedComponents.customizedEvalHook import customLossEval, customEvalHook

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        with open(os.path.join(cfg.train.output_dir, "ap_steel_val.json"), "a") as f:
            json.dump(ret, f)
            f.write("\n")
        print_csv_format(ret)
        return ret

def do_source_test(cfg, model):
    if "evaluator_source" in cfg.dataloader and "test_source" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test_source), instantiate(cfg.dataloader.evaluator_source)
        )
        with open(os.path.join(cfg.train.output_dir, "ap_steel_test.json"), "a") as f:
            json.dump(ret, f)
            f.write("\n")
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

    for param in model.backbone.parameters():
        param.requires_grad = False

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (customAMPTrainer if cfg.train.amp.enabled else customSimpleTrainer)(model, 
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
            hooks.EvalHook(cfg.train.eval_period, lambda: do_source_test(cfg, model)),

            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
            # customEvalHook(cfg.train.log_period * 5, lambda: do_validation_loss(cfg, model))
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

    # cfg.train.max_iter = 800
    # cfg.train.checkpointer.period = 40
    # cfg.train.eval_period = 80
    # cfg.train.log_period = 10
    cfg.train.max_iter = 60000
    cfg.train.checkpointer.period = 40
    cfg.train.eval_period = 400
    cfg.train.log_period = 20
    cfg.optimizer.lr = 0.001
    cfg.lr_multiplier = LazyCall(WarmupParamScheduler)(
        scheduler = LazyCall(MultiStepParamScheduler)(
            # values=[1.0, 0.81, 0.73, 0.4, 0.1],
            # milestones=[25000, 35000, 45000, 50000],
            # values=[1.0, 0.81, 0.73, 0.65, 0.5],
            # milestones=[25000, 35000, 40000, 45000, 50000],
            # values = [1.0, 0.9, 0.1, 0.01],
            # milestones=[25000, 50000, 60000, 70000],
            # values = [1.0, 0.1, 0.01, 0.005],
            # milestones=[45000, 50000, 60000, 70000],
            values=[1.0, 0.81, 0.73, 0.65, 0.3, 0.1, 0.01],
            milestones=[25000, 35000, 40000, 45000, 50000, 55000, 60000],
        ),
        warmup_length= 0.1,
        warmup_method="linear",
        warmup_factor=0.001,
    )
    if args.num_gpus > 1:
        cfg.model.backbone.bottom_up.stem.norm = \
        cfg.model.backbone.bottom_up.stages.norm = "SyncBN"
        cfg.model.backbone.norm = "SyncBN"
    else:
        cfg.model.backbone.bottom_up.stem.norm = \
        cfg.model.backbone.bottom_up.stages.norm = "BN"
        cfg.model.backbone.norm = "BN"
    
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
