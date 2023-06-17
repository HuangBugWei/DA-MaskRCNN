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

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import collections

def do_tsne(cfg, model, dataloader1, dataloader2):
    model.eval()
    with torch.no_grad():
        res3 = None
        res4 = None
        res5 = None
        
        for idx, (batch1, batch2) in enumerate(zip(instantiate(dataloader1), instantiate(dataloader2))):
            # print("---------------size---------------")
            # print(batch1[0]["image"].size())
            # print(batch2[0]["image"].size())
            # images1 = model.preprocess_image(batch1)
            # images2 = model.preprocess_image(batch2)
            # print(images1.tensor.size())
            # print(images2.tensor.size())
            # images = model.preprocess_image(batch1 + batch2)
            # print(images.tensor.size())
            # break
            images = model.preprocess_image(batch1 + batch2)
            features, res = model.backbone(images.tensor)
            
            # print(res["res3"].size())
            if res3 is not None:
                res3 = np.concatenate((res3, res["res3"].cpu().detach().numpy()), axis=0)
            else:
                res3 = res["res3"].cpu().detach().numpy()
            
            if res4 is not None:
                res4 = np.concatenate((res4, res["res4"].cpu().detach().numpy()), axis=0)
            else:
                res4 = res["res4"].cpu().detach().numpy()

            if res5 is not None:
                res5 = np.concatenate((res5, res["res5"].cpu().detach().numpy()), axis=0)
            else:
                res5 = res["res5"].cpu().detach().numpy()
            
            if res3.shape[0] > 200:
                break
        
        print("-------------------")
        print(res3.shape)
        print(res4.shape)
        print(res5.shape)
    
    res3 = res3.reshape(res3.shape[0], -1)
    res4 = res4.reshape(res4.shape[0], -1)
    res5 = res5.reshape(res5.shape[0], -1)

    tsne3 = TSNE(n_components=2).fit_transform(res3)
    tsne4 = TSNE(n_components=2).fit_transform(res4)
    tsne5 = TSNE(n_components=2).fit_transform(res5)

    print(tsne3.shape)
    print(tsne4.shape)
    print(tsne5.shape)

    plt.figure(figsize=(10,10))

    for i in np.arange(len(tsne3)):
        if (i // 8) % 2:
            c = "b"
            label = "target"
        else:
            c = "y"
            label = "source"
        x, y = (tsne3[i,0],tsne3[i,1])
        plt.scatter(x,y,c = c, label=label)
    # plt.legend()
    plt.savefig("tSNE-res3.png")

    for i in np.arange(len(tsne4)):
        if (i // 8) % 2:
            c = "b"
            label = "target"
        else:
            c = "y"
            label = "source"
        x, y = (tsne4[i,0],tsne4[i,1])
        plt.scatter(x,y,c = c, label=label)
    # plt.legend()
    plt.savefig("tSNE-res4.png")

    for i in np.arange(len(tsne5)):
        if (i // 8) % 2:
            c = "b"
            label = "target"
        else:
            c = "y"
            label = "source"
        x, y = (tsne5[i,0],tsne5[i,1])
        plt.scatter(x,y,c = c, label=label)
    # plt.legend()
    plt.savefig("tSNE-res5.png")


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    
    DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-dataset", txt=False))
    DatasetCatalog.register('steel_test_source', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/rebar-revit-auto-test.txt", txt=True))
    
    MetadataCatalog.get("steel_test").set(thing_classes=['intersection', 'spacing'])
    MetadataCatalog.get("steel_test_source").set(thing_classes=['intersection', 'spacing'])
    
    # cfg.model.roi_heads.num_classes = 2
    cfg.train.output_dir = "./output"

    cfg.dataloader.test = LazyCall(build_detection_test_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names="steel_test", filter_empty=False),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=False,
                                augmentations=[
                                    LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                                ],
                                image_format="${...train.mapper.image_format}",
                            ),
                            batch_size = 8,
                            num_workers = 4,
                            )
    cfg.dataloader.test_source = LazyCall(build_detection_test_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names="steel_test_source", filter_empty=False),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=True,
                                use_instance_mask=True,
                                augmentations=[
                                    LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                                ],
                                image_format="${...train.mapper.image_format}",
                            ),
                            batch_size = 8,
                            num_workers = 4,
                            )
    
    dataloader_dict = collections.defaultdict(list)
    dataloader_dict["source"].append(cfg.dataloader.test_source)
    dataloader_dict["target"].append(cfg.dataloader.test)

    cfg.train.init_checkpoint = "./output/model_final.pth"
    default_setup(cfg, args)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

    print(do_tsne(cfg, model, cfg.dataloader.test_source, cfg.dataloader.test))
    


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    
    
    launch(
        main,
        1,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )