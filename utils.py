import os, glob, json
import numpy as np
from PIL import Image
from detectron2.structures import BoxMode

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

def get_no_label_dicts(img_dir):
    """
    img_dir: the folder with the iamges in it, 
    will return Detectron2 standard dataset dicts list[dict]
    with the fields that need for common tasks, (file_name, height, width, image_id)
    ref: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
    """
    dataset_dicts = []
    folder = os.path.abspath(img_dir)
    
    for idx, img_file_name in enumerate(os.listdir(folder)):
        
        record = {}
        
        filename = os.path.join(folder, img_file_name)
        try:
            image = Image.open(filename)
        
            record["file_name"] = filename
            record["image_id"] = idx
            record["width"], record["height"] = image.size
            dataset_dicts.append(record)
        except:
            print(f"{filename} is not image.")

    return dataset_dicts