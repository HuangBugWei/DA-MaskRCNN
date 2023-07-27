import torch, torchvision
from torch import Tensor
import argparse, os, cv2
from PIL import Image
import numpy as np
import pickle
import detectron2.data.transforms as T
from detectron2.config import LazyConfig, instantiate, LazyCall
def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type = str, required = True)
    # parser.add_argument("-m", "--model", type = str, required = True)
    parser.add_argument("-o", "--output", type = str, required = True)
    parser.add_argument("--num-gpus", type = int, default = 1)
    parser.add_argument("--thres", type = float, default = 0.5)
    # parser.add_argument("--config-file", required = True, metavar="FILE", help="path to config file")

    return parser.parse_args()

def single_image_process(im, augs, model):
    height, width = im.shape[:2]
    print("og")
    print(height, width)
    height = torch.tensor(height, dtype=torch.int8)
    width = torch.tensor(width, dtype=torch.int8)
    with torch.no_grad():
        image = augs.get_transform(im).apply_image(im)
        print("test")
        print(image.shape[0], image.shape[1])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        input = {"image": image, "height": height, "width": width}
        # input = {"image": get_sample_coco_image()}
        inputs = [input]
        predictions = model(inputs)[0]
        # print(type(predictions))
        # with open("ts-result.pkl", "wb") as f:
        #     pickle.dump(predictions, f)
        # predictions = model.inference(inputs, do_postprocess=False)[0]
        # print(predictions.pred_classes)
        # print(predictions.pred_masks)
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
    
    model = torch.jit.load("/home/aicenter/DA-MaskRCNN/torchscript/script/model.ts")
    
    model.to("cpu")
    
    if os.path.isfile(args.input):
        filename = args.input
        im = cv2.imread(filename)
        augs = instantiate(LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333))
        if im is not None:

            predictions = single_image_process(im, augs, model)
            
            ret = create_binary_mask(predictions["pred_masks"],
                                     predictions["pred_classes"])
            basename = os.path.splitext(os.path.basename(filename))[0]
            path = os.path.join(args.output, f"pred-{str(args.thres)}-" + basename + "-seg.jpg")
            print(path)
            draw_seg_result(ret, path)
            # vis = Visualizer(im[:, :, ::-1], metadata, scale=1)
            path = os.path.join(args.output, f"pred-{str(args.thres)}" + basename + "-ins.jpg")
            # vis.draw_instance_predictions(predictions["instances"].to("cpu")).save(path)

if __name__ == "__main__":
    args = parser()
    main(args)