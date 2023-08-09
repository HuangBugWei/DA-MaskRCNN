from DAMaskRCNN import model, train, dataloader
from DAMaskRCNN import SGD as optimizer
# from DAMaskRCNN import Adam as optimizer

from detectron2.data import MetadataCatalog, DatasetCatalog
from utils import *
from detectron2.config import LazyCall
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator

model.backbone.bottom_up.freeze_at = 0
model.roi_heads.num_classes = 2
model.roi_heads.box_predictor.test_score_thresh = 0.5
train["init_checkpoint"] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train["output_dir"] = "./ablation-DA-25k-1k"
model.do_domain = True

DatasetCatalog.register('steel_train', lambda : get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/valid-1k-10-types.txt", txt=True))
# DatasetCatalog.register('steel_train_target', lambda : get_no_label_dicts("/home/aicenter/maskrcnn/rebar-target-dataset/imgs"))
# DatasetCatalog.register('steel_train_target', lambda : get_no_label_dicts("/home/aicenter/maskrcnn/rebar-target-dataset/da-train-target.txt", txt=True))
DatasetCatalog.register('steel_train_target', lambda : get_no_label_dicts("/home/aicenter/maskrcnn/rebar-target-dataset/da-train-target.txt", txt=True))
DatasetCatalog.register('steel_test', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/target-val.txt", txt=True))
DatasetCatalog.register('steel_test_source', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-labeled-test-dataset/target-slab.txt", txt=True))
# DatasetCatalog.register('steel_test_source', lambda :  get_rebar_dicts("/home/aicenter/maskrcnn/rebar-revit-auto-dataset/rebar-revit-auto-test.txt", txt=True))

MetadataCatalog.get("steel_train").set(thing_classes=['intersection', 'spacing'])
MetadataCatalog.get("steel_test").set(thing_classes=['intersection', 'spacing'])
MetadataCatalog.get("steel_test_source").set(thing_classes=['intersection', 'spacing'])

dataloader.train = LazyCall(build_detection_train_loader)(
                    dataset=LazyCall(get_detection_dataset_dicts)(names="steel_train"),
                    mapper=LazyCall(DatasetMapper)(
                        is_train=True,
                        augmentations=[
                            LazyCall(T.RandomBrightness)(intensity_min=0.4, intensity_max=1.6),
                            LazyCall(T.RandomContrast)(intensity_min=0.4, intensity_max=1.6),
                            #LazyCall(T.RandomRotation)(angle=[-5, 5], expand=False, center=None, sample_style='range', interp=2),
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
                    total_batch_size=6,
                    num_workers=4,
                    )
dataloader.train_target = LazyCall(build_detection_train_loader)(
                            dataset=LazyCall(get_detection_dataset_dicts)(names="steel_train_target"),
                            mapper=LazyCall(DatasetMapper)(
                                is_train=True,
                                augmentations=[
                                    LazyCall(T.RandomBrightness)(intensity_min=0.4, intensity_max=1.6),
                                    LazyCall(T.RandomContrast)(intensity_min=0.4, intensity_max=1.6),
                                    LazyCall(T.ResizeShortestEdge)(
                                        short_edge_length=720,
                                        sample_style="choice",
                                        max_size=1280,
                                    ),
                                    LazyCall(T.RandomCrop)(crop_type="absolute", 
                                                           crop_size=(720, 960)),
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
                            total_batch_size=6,
                            num_workers=4,
                            )

dataloader.test = LazyCall(build_detection_test_loader)(
                    dataset=LazyCall(get_detection_dataset_dicts)(names="steel_test", filter_empty=False),
                    mapper=LazyCall(DatasetMapper)(
                        is_train=True,
                        use_instance_mask=True,
                        augmentations=[
                            LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                        ],
                        image_format="${...train.mapper.image_format}",
                    ),
                    num_workers=4,
                    )
dataloader.test_source = LazyCall(build_detection_test_loader)(
                        dataset=LazyCall(get_detection_dataset_dicts)(names="steel_test_source", filter_empty=False),
                        mapper=LazyCall(DatasetMapper)(
                            is_train=True,
                            use_instance_mask=True,
                            augmentations=[
                                LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
                            ],
                            image_format="${...train.mapper.image_format}",
                        ),
                        num_workers=4,
                        )
dataloader.evaluator = LazyCall(COCOEvaluator)(
                                dataset_name="${..test.dataset.names}",
                                output_dir=train["output_dir"]
                            )
dataloader.evaluator_source = LazyCall(COCOEvaluator)(
                                dataset_name="${..test_source.dataset.names}",
                                output_dir=train["output_dir"]
                            )