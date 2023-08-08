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

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
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

    # DatasetCatalog.register('steel_train', lambda :  get_rebar_dicts("./rebar-revit-auto-dataset/files-7000.txt", txt=True))
    # DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("./rebar-dataset", txt=False))
    DatasetCatalog.register('steel_train', lambda :  get_rebar_dicts("./rebar-revit-auto-dataset/slabs.txt", txt=True))
    DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("./rebar-revit-auto-dataset/B.txt", txt=True))
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
    cfg.dataloader.train.dataset = LazyCall(get_detection_dataset_dicts)(names="steel_train")
    cfg.dataloader.train.total_batch_size = 12
    cfg.train.output_dir = "./trial"
    cfg.train.max_iter = 50
    cfg.train.checkpointer.period = 2000
    cfg.train.eval_period = 1000
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

def get_rebar_dicts(img_dir, txt = True):
    dataset_dicts = []
    folder = os.path.dirname(os.path.abspath(img_dir))
    jsonFolder = []
    if txt:
        with open(os.path.abspath(img_dir), "r") as f:
            for idx, line in enumerate(f):
                jsonFolder.append(os.path.join(folder, "json", line.rstrip()))
    else:
        jsonFolder = glob.glob(os.path.join(img_dir, "json", "*.json"))
        folder = os.path.abspath(img_dir)
    # for idx, json_file in enumerate(glob.glob(os.path.join(img_dir, "json", "*.json"))):
    for idx, json_file in enumerate(jsonFolder):
        
        with open(json_file) as f:
            imgs_anns = json.load(f)

        record = {}
        
        filename = os.path.join(folder, "imgs", os.path.basename(imgs_anns["imagePath"]))
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = imgs_anns["imageHeight"]
        record["width"] = imgs_anns["imageWidth"]
      
        annos = imgs_anns["shapes"]
        # annos: list[dict]
        
        objs = []
        for anno in annos:
            # anno: dict
            
            # fix some data may have wrong label, such as polygon with only 2 or 1 points...
            # triangle shape also is not reasonable to our task so I set the threshold at 4
            if len(anno["points"]) < 4:
                continue
            
            px = [pair[0] for pair in anno["points"]]
            py = [pair[1] for pair in anno["points"]]
            poly = [p for x in anno["points"] for p in x]
            if anno["label"] == "intersection":
                cls = 0
            if anno["label"] == "spacing":
                cls = 1

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": cls,
                "iscrowd": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        
        dataset_dicts.append(record)
    return dataset_dicts




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