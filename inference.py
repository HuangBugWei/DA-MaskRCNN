import os, time
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate, LazyCall

from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.transforms as T

from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)

import cv2, torch, argparse
from detectron2.utils.visualizer import Visualizer
from utils import get_no_label_dicts
from PIL import Image
# import pickle

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type = str, required = True)
    parser.add_argument("-m", "--model", type = str, required = True)
    parser.add_argument("-o", "--output", type = str, required = True)
    parser.add_argument("--num-gpus", type = int, default = 1)
    parser.add_argument("--thres", type = float, default = 0.5)
    parser.add_argument("--config-file", required = True, metavar="FILE", help="path to config file")

    return parser.parse_args()

def single_image_process(im, augs, model):
    height, width = im.shape[:2]
    with torch.no_grad():
        image = augs.get_transform(im).apply_image(im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        
        predictions = model([inputs])[0]
        # print(type(predictions))
        # with open("og-result.pkl", "wb") as f:
        #     pickle.dump(predictions, f)
        # print(predictions["instances"].pred_classes.shape)
        # print(predictions["instances"].pred_masks.shape)
    return predictions


def create_binary_mask(masks, classes):
    ret = []
    # for i in torch.unique(classes, sorted=True):
    for i in range(2):
        ret.append(torch.zeros(masks.shape[1:], dtype=torch.bool))
    for cls, mask in zip(classes, masks):
        ret[cls.item()] = torch.logical_or(ret[cls.item()], mask)
    
    return ret

def draw_result(inputs, outputs, outputPath, metadata):
    for input, output in zip(inputs, outputs):
        im = cv2.imread(input["file_name"])
        if im.shape[0] > 2000:
            vis = Visualizer(im[:, :, ::-1], metadata, scale=0.5)
        else:
            vis = Visualizer(im[:, :, ::-1], metadata, scale=1)
        basename = os.path.splitext(os.path.basename(input["file_name"]))[0]
        path = os.path.join(outputPath, basename + "-ins.jpg")
        vis.draw_instance_predictions(output["instances"].to("cpu")).save(path)
        ret = create_binary_mask(output["instances"].pred_masks.to("cpu"),
                                 output["instances"].pred_classes.to("cpu"),)
        path = os.path.join(outputPath, basename + "-seg.jpg")
        draw_seg_result(ret, path)

def draw_seg_result(ret, outputPath):
    r = torch.zeros(ret[0].shape, dtype=torch.uint8)
    g = torch.zeros(ret[0].shape, dtype=torch.uint8)
    b = torch.zeros(ret[0].shape, dtype=torch.uint8)
    r = torch.where(ret[0], 255, 0)
    b = torch.where(ret[1], 255, 0)
    r = torch.unsqueeze(r, 0)
    g = torch.unsqueeze(g, 0)
    b = torch.unsqueeze(b, 0)
    
    mix = torch.cat((r, g, b), axis=0)
    mix = torch.permute(mix, (1, 2, 0))
    if mix.shape[0] > 2000:
        Image.fromarray(np.uint8(mix)).resize(
            (int(mix.shape[1]*0.5), int(mix.shape[0]*0.5))).save(outputPath)
    else:
        Image.fromarray(np.uint8(mix)).save(outputPath)

def main(args):
    cfg = LazyConfig.load(args.config_file)
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
    # model.to(cfg.train.device)
    model.to("cuda")

    cfg.train.init_checkpoint = args.model
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    augs = instantiate(LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333))
    model.eval()
    os.makedirs(args.output, exist_ok = True)
    os.makedirs(os.path.join(args.output, str(args.thres)), exist_ok = True)
    
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    
    MetadataCatalog.get("steel_infer").set(thing_classes=['intersection', 'spacing'])
    metadata = MetadataCatalog.get('steel_infer')
    
    inferList = []
    if os.path.isfile(args.input):
        filename = args.input
        im = cv2.imread(filename)
        if im is not None:
            record = []
            for i in range(100):
                start = time.time()
                predictions = single_image_process(im, augs, model)
                span = time.time() - start
                record.append(span)
            print(sum(record)/len(record))
            # torch.save(predictions, "./postprocessing/sample-result.pt")
            ret = create_binary_mask(predictions["instances"].pred_masks,
                                     predictions["instances"].pred_classes)
            basename = os.path.splitext(os.path.basename(filename))[0]
            path = os.path.join(args.output, f"pred-{str(args.thres)}-" + basename + "-seg.jpg")
            draw_seg_result(ret, path)
            vis = Visualizer(im[:, :, ::-1], metadata, scale=1)
            path = os.path.join(args.output, f"pred-{str(args.thres)}" + basename + "-ins.jpg")
            vis.draw_instance_predictions(predictions["instances"].to("cpu")).save(path)

    elif os.path.isdir(args.input):
        # DatasetCatalog.register('steel_infer', lambda : get_no_label_dicts("/home/aicenter/maskrcnn/rebar-target-dataset/draw.txt", True))
        # DatasetCatalog.register('steel_infer', lambda : get_no_label_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/draw-val-slab.txt", True))
        # DatasetCatalog.register('steel_infer', lambda : get_no_label_dicts("/home/aicenter/DA-MaskRCNN/online-steel-image/log.txt", True))
        DatasetCatalog.register('steel_infer', lambda : get_no_label_dicts("/home/aicenter/DA-MaskRCNN/online-steel-image-crop/log.txt", True))
        dataloader = instantiate(LazyCall(build_detection_test_loader)(
                        dataset=LazyCall(get_detection_dataset_dicts)(names="steel_infer", filter_empty=False),
                        mapper=LazyCall(DatasetMapper)(
                            is_train = False,
                            augmentations=[
                                LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                            ],
                            image_format="BGR",
                        ),
                        batch_size = 1,
                        num_workers=4,
                        ))
        outputPath = os.path.join(args.output, str(args.thres))
        for idx, inputs in enumerate(dataloader):
            if (idx + 1) % 1000 == 0:
                time.sleep(5)
                print("wait for buffering")

            outputs = model(inputs)
            draw_result(inputs, outputs, outputPath, metadata)
            
    else:
        print("error input")
        return
    

    # print(predictions)
    # print(predictions["instances"].pred_classes)
    # print(predictions["instances"].pred_masks.shape)
    # vis = Visualizer(im[:, :, ::-1], metadata, scale=1)
    # vis_pred = vis.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
    # # Image.fromarray(vis_pred).show()
    # Image.fromarray(vis_pred).save(os.path.join(args.output, "vanilla-result.png"))
    # # vis_pred = vis.draw_binary_mask(predictions["instances"].pred_masks[0].to("cpu").numpy()).get_image()
    # # vis_pred = vis.draw_binary_mask(predictions["instances"].pred_masks[1].to("cpu").numpy()).get_image()
    # # Image.fromarray(vis_pred).show()



if __name__ == "__main__":
    args = parser()
    main(args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
