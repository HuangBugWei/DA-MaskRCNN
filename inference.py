import os, time
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate, LazyCall
# from detectron2.engine import (
#     AMPTrainer,
#     SimpleTrainer,
#     default_argument_parser,
#     default_setup,
#     default_writers,
#     hooks,
#     launch,
# )

from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.transforms as T
# from detectron2.structures import BoxMode
# from detectron2.evaluation import COCOEvaluator

from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)

import cv2, torch, argparse
from detectron2.utils.visualizer import Visualizer
from utils import get_no_label_dicts

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type = str, required = True)
    parser.add_argument("-m", "--model", type = str, required = True)
    parser.add_argument("-o", "--output", type = str, required = True)
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

        return predictions


def create_binary_mask(masks, classes):
    masks.shape[1:]

def draw_result(inputs, outputs, outputPath, metadata):
    for input, output in zip(inputs, outputs):
        im = cv2.imread(input["file_name"])
        if im.shape[0] > 2000:
            vis = Visualizer(im[:, :, ::-1], metadata, scale=0.5)
        else:
            vis = Visualizer(im[:, :, ::-1], metadata, scale=1)
        basename = os.path.splitext(os.path.basename(input["file_name"]))[0]
        path = os.path.join(outputPath, "pred-" + basename + ".jpg")
        vis.draw_instance_predictions(output["instances"].to("cpu")).save(path)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg.model.roi_heads.box_predictor.test_score_thresh = args.thres
    model = instantiate(cfg.model)
    # model.to(cfg.train.device)
    model.to("cpu")

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
            predictions = single_image_process(im, augs, model)
            vis = Visualizer(im[:, :, ::-1], metadata, scale=1)
            basename = os.path.splitext(os.path.basename(filename))[0]
            path = os.path.join(args.output, str(args.thres), "pred-" + basename + ".jpg")
            vis.draw_instance_predictions(predictions["instances"].to("cpu")).save(path)
    elif os.path.isdir(args.input):
        DatasetCatalog.register('steel_infer', lambda :  get_no_label_dicts("/home/aicenter/maskrcnn/rebar-target-dataset/draw.txt", True))
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