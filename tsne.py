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
import logging, os, json, glob, argparse
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
import torch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import collections, time, copy

def do_tsne(x):
    result = TSNE(n_components=2, 
                 perplexity=50,
                 learning_rate="auto",
                 n_iter=500,
                 init="pca",
                 ).fit_transform(x)
    return result

def do_pca(x):
    result = PCA(n_components=2).fit_transform(x)
    return result

def do_analyze(cfg, model, dataloader1, dataloader2, suffix, pca=False):
    
    model.eval()
    with torch.no_grad():
        res3 = []
        res4 = []
        res5 = []
        
        # for idx, (batch1, batch2) in enumerate(zip(dataloader1, dataloader2)):
            
        #     if idx == 0:
        #         print(batch1[0]["file_name"])
        #         print(batch2[0]["file_name"])
    
        #     # print("---------------size---------------")
        #     # print(batch1[0]["image"].size())
        #     # print(batch2[0]["image"].size())
        #     # images1 = model.preprocess_image(batch1)
        #     # images2 = model.preprocess_image(batch2)
        #     # print(images1.tensor.size())
        #     # print(images2.tensor.size())
        #     # images = model.preprocess_image(batch1 + batch2)
        #     # print(images.tensor.size())
        #     # break

        #     images = model.preprocess_image(batch1 + batch2)
        #     features, res = model.backbone(images.tensor)
            
        #     res3.append(res["res3"].cpu().detach())
        #     res4.append(res["res4"].cpu().detach())
        #     res5.append(res["res5"].cpu().detach())
        #     del res["res3"]
        #     del res["res4"]
        #     del res["res5"]

        #     if len(res3) * 16 > 1000:
        #         break
        
        for idx, batch1 in enumerate(dataloader1):
            images = model.preprocess_image(batch1)
            features, res = model.backbone(images.tensor)
            
            res3.append(res["res3"].cpu().detach())
            res4.append(res["res4"].cpu().detach())
            res5.append(res["res5"].cpu().detach())
            del res["res3"]
            del res["res4"]
            del res["res5"]

            # if idx * 8 > 20:
            #     break
        
        for idx, batch2 in enumerate(dataloader2):
            images = model.preprocess_image(batch2)
            features, res = model.backbone(images.tensor)
            
            res3.append(res["res3"].cpu().detach())
            res4.append(res["res4"].cpu().detach())
            res5.append(res["res5"].cpu().detach())
            del res["res3"]
            del res["res4"]
            del res["res5"]

            # if idx * 8 > 20:
            #     break
        
        res3 = torch.cat(res3, axis = 0).numpy()
        res4 = torch.cat(res4, axis = 0).numpy()
        res5 = torch.cat(res5, axis = 0).numpy()
        
        print("-------------------")
        print(res3.shape)
        print(res4.shape)
        print(res5.shape)
    
    res3 = res3.reshape(res3.shape[0], -1)
    res4 = res4.reshape(res4.shape[0], -1)
    res5 = res5.reshape(res5.shape[0], -1)
    print("-------------------")
    print(res3.shape)
    print(res4.shape)
    print(res5.shape)

    if pca:
        ret3 = do_pca(res3)
        ret4 = do_pca(res4)
        ret5 = do_pca(res5)
    else:
        ret3 = do_tsne(res3)
        ret4 = do_tsne(res4)
        ret5 = do_tsne(res5)

    print(ret3.shape)
    print(ret4.shape)
    print(ret5.shape)
    half_idx = ret3.shape[0] // 2
    plt.figure(figsize=(5, 5))

    # for i in np.arange(len(ret3)):
    #     if (i // 8) % 2:
    #         c = "b"
    #         label = "target"
    #     else:
    #         c = "y"
    #         label = "source"
    #     x, y = (ret3[i,0],ret3[i,1])
    #     plt.scatter(x, y, c=c, label=label)
    
    plt.scatter(ret3[:half_idx, 0], ret3[:half_idx, 1], c="blue",
                alpha=0.8, edgecolors='none', label="source")
    plt.scatter(ret3[half_idx:, 0], ret3[half_idx:, 1], c="green",
                alpha=0.8, edgecolors='none', label="target")
    plt.legend()
    plt.axis("off")
    if pca:
        plt.title(f"PCA-res3-{suffix}")
        plt.savefig(os.path.join(cfg.train.output_dir, f"PCA-res3-{suffix}.png"))
    else:
        plt.title(f"tSNE-res3-{suffix}")
        plt.savefig(os.path.join(cfg.train.output_dir, f"tSNE-res3-{suffix}.png"))
    plt.clf()

    plt.scatter(ret4[:half_idx, 0], ret4[:half_idx, 1], c="blue",
                alpha=0.8, edgecolors='none', label="source")
    plt.scatter(ret4[half_idx:, 0], ret4[half_idx:, 1], c="green",
                alpha=0.8, edgecolors='none', label="target")
    plt.legend()
    plt.axis("off")
    if pca:
        plt.title(f"PCA-res4-{suffix}")
        plt.savefig(os.path.join(cfg.train.output_dir, f"PCA-res4-{suffix}.png"))
    else:
        plt.title(f"tSNE-res4-{suffix}")
        plt.savefig(os.path.join(cfg.train.output_dir, f"tSNE-res4-{suffix}.png"))
    plt.clf()

    plt.scatter(ret5[:half_idx, 0], ret5[:half_idx, 1], c="blue",
                alpha=0.8, edgecolors='none', label="source")
    plt.scatter(ret5[half_idx:, 0], ret5[half_idx:, 1], c="green",
                alpha=0.8, edgecolors='none', label="target")
    plt.legend()
    plt.axis("off")
    if pca:
        plt.title(f"PCA-res5-{suffix}")
        plt.savefig(os.path.join(cfg.train.output_dir, f"PCA-res5-{suffix}.png"))
    else:
        plt.title(f"tSNE-res5-{suffix}")
        plt.savefig(os.path.join(cfg.train.output_dir, f"tSNE-res5-{suffix}.png"))
    plt.clf()

def main(args):
    cfg = LazyConfig.load(args.config_file)
    # cfg = LazyConfig.apply_overrides(cfg, args.opts)

    DatasetCatalog.clear()
    MetadataCatalog.clear()

    DatasetCatalog.register('steel_test', lambda : get_no_label_dicts("/home/aicenter/maskrcnn/rebar-target-dataset/feature-analysis.txt", txt=True))
    DatasetCatalog.register('steel_test_source', lambda : get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/feature-analysis.txt", txt=True))
    # DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-dataset", txt=False))
    # DatasetCatalog.register('steel_test_source', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/rebar-revit-auto-test.txt", txt=True))
    
    MetadataCatalog.get("steel_test").set(thing_classes=['intersection', 'spacing'])
    MetadataCatalog.get("steel_test_source").set(thing_classes=['intersection', 'spacing'])
    
    # cfg.model.roi_heads.num_classes = 2
    cfg.train.output_dir = args.input

    cfg.dataloader.test = LazyCall(build_detection_test_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names="steel_test", filter_empty=False),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=False,
                                augmentations=[
                                    LazyCall(T.ResizeShortestEdge)(
                                        short_edge_length=720,
                                        sample_style="choice",
                                        max_size=1280,
                                    ),
                                    LazyCall(T.RandomCrop)(crop_type="absolute", 
                                                           crop_size=(720, 960)),
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
    
    # dataloader_dict = collections.defaultdict(list)
    # dataloader_dict["source"].append(cfg.dataloader.test_source)
    # dataloader_dict["target"].append(cfg.dataloader.test)

    # cfg.train.init_checkpoint = "/home/aicenter/DA-MaskRCNN/ablation-DA-25k/model_final.pth"
    # # default_setup(cfg, args)
    if args.num_gpus > 1:
        cfg.model.backbone.bottom_up.stem.norm = \
        cfg.model.backbone.bottom_up.stages.norm = "SyncBN"
        cfg.model.backbone.norm = "SyncBN"
    else:
        cfg.model.backbone.bottom_up.stem.norm = \
        cfg.model.backbone.bottom_up.stages.norm = "BN"
        cfg.model.backbone.norm = "BN"
    
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    
    # model = create_ddp_model(model)
    # DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    start = time.time()
    dataloader1, dataloader2 = instantiate(cfg.dataloader.test_source), instantiate(cfg.dataloader.test)
    print(f"instantiate dataloader time {time.time() - start} s")
    if args.model is not None:
        iter = os.path.splitext(os.path.basename(args.model))[0].split("_")[1]
        cfg.train.init_checkpoint = os.path.join(cfg.train.output_dir, args.model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        start = time.time()
        print(do_analyze(cfg,
                        model, 
                        copy.deepcopy(dataloader1),
                        copy.deepcopy(dataloader2),
                        f"-{args.suffix}-{iter}",
                        args.pca),
                        )
        print(f"process time {time.time() - start} s")
        return

    for ckpt in sorted(os.listdir(cfg.train.output_dir)):
        if ckpt.startswith("model") and ckpt.endswith(".pth"):
            print(f"now loading {ckpt}")
            iter = os.path.splitext(os.path.basename(ckpt))[0].split("_")[1]
            if iter == "final":
                break
            print(f"now iter {iter}")
            cfg.train.init_checkpoint = os.path.join(cfg.train.output_dir, ckpt)
            DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
            start = time.time()
            print(do_analyze(cfg,
                          model,
                          copy.deepcopy(dataloader1),
                          copy.deepcopy(dataloader2),
                          f"-{args.suffix}-{iter}",
                          args.pca),
                          )
            print(f"process time {time.time() - start} s")
    
def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type = str, required = True)
    parser.add_argument("-m", "--model", type = str)
    parser.add_argument("-s", "--suffix", type = str, required = True)
    parser.add_argument("--pca", action='store_true', help="if use this flag, use pca to analyze, else use tSNE.")
    parser.add_argument("--num-gpus", type = int, default = 1)
    parser.add_argument("--config-file", required = True, metavar="FILE", help="path to config file")

    return parser.parse_args()

if __name__ == "__main__":
    args = parser()
    # print(args.pca)
    start = time.time()
    main(args)
    print(f"total time {time.time() - start} s")
    # args = default_argument_parser().parse_args()
    
    # launch(
    #     main,
    #     1,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )