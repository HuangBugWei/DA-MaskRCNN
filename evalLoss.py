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
from customizedComponents.customizedTrainer import customAMPTrainer, customSimpleTrainer
import torch, argparse, copy, time
from customizedComponents.customizedEvalHook import customLossEval, customEvalHook

logger = logging.getLogger("detectron2")

def build_loader(cfg, dataset_name):
    return instantiate(LazyCall(build_detection_test_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names=dataset_name, filter_empty=True),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=True,
                                use_instance_mask=True,
                                augmentations=[
                                    LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                                ],
                                image_format="BGR",
                            ),
                            num_workers=4,
                            ))

def main(args):
    cfg = LazyConfig.load(args.config_file)

    DatasetCatalog.clear()
    MetadataCatalog.clear()


    if args.bim:
        DatasetCatalog.register('steel_test_bim', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/files-val-200.txt", txt = True))
        MetadataCatalog.get("steel_test_bim").set(thing_classes=['intersection', 'spacing'])
        bimloader = build_loader(cfg, "steel_test_bim")
    else:
        DatasetCatalog.register('steel_val', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/target-val.txt", txt = True))
        MetadataCatalog.get("steel_val").set(thing_classes=['intersection', 'spacing'])
        valloader = build_loader(cfg, "steel_val")
    
        
    cfg.model.roi_heads.num_classes = 2
    cfg.train.output_dir = args.input
    cfg.model.roi_heads.box_predictor.test_score_thresh = args.thres

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
    model = create_ddp_model(model)
    # model.to("cpu")
    
    for ckpt in sorted(os.listdir(cfg.train.output_dir)):
        if ckpt.startswith("model") and ckpt.endswith(".pth"):
            print(f"now loading {ckpt}")
            iter = os.path.splitext(os.path.basename(ckpt))[0].split("_")[1]
            if iter == "final":
                break
            print(f"now iter {iter}")
            cfg.train.init_checkpoint = os.path.join(cfg.train.output_dir, ckpt)
            DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
            
            
            if args.bim:
                print("bim-ap")
                losses = customLossEval(model, copy.deepcopy(bimloader), False)
                total_loss = 0
                for value in losses.values():
                    total_loss += value
                losses["total_loss"] = total_loss
                losses["iter"] = iter
                with open(os.path.join(cfg.train.output_dir, f"bim-val-loss.json"), "a") as f:
                    json.dump(losses, f)
                    f.write("\n")
                print(losses)
            else:
                print("val-ap")
                losses = customLossEval(model, copy.deepcopy(valloader), False)
                total_loss = 0
                for value in losses.values():
                    total_loss += value
                losses["total_loss"] = total_loss
                losses["iter"] = iter
                with open(os.path.join(cfg.train.output_dir, f"steel-val-loss.json"), "a") as f:
                    json.dump(losses, f)
                    f.write("\n")
                print(losses)
    
def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type = str, required = True)
    parser.add_argument("-b", "--bim", action='store_true')
    parser.add_argument("--num-gpus", type = int, default = 1)
    parser.add_argument("--thres", type = float, default = 0.5)
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
