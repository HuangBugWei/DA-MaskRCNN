from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

from customizedComponents.customizedFPN import customFPN
from customizedComponents.customizedRPN import customRPN
from customizedComponents.customizedRCNN import customRCNN
from customizedComponents.customizedROIHeads import customROIHeads
from customizedComponents.customizedMaskHead import customMaskHead

import torch

from detectron2.solver.build import get_default_optimizer_params

from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.solver import WarmupParamScheduler
from omegaconf import OmegaConf

import detectron2.data.transforms as T

from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)


def default_X_scheduler(num_X):
    """
    Returns the config for a default multi-step LR scheduler such as "1x", "3x",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed twice at the end of training
    following the strategy defined in "Rethinking ImageNet Pretraining", Sec 4.

    Args:
        num_X: a positive real number

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = num_X * 90000

    if num_X <= 2:
        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            # note that scheduler is scale-invariant. This is equivalent to
            # milestones=[6, 8, 9]
            milestones=[60000, 80000, 90000],
        )
    else:
        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=[total_steps_16bs - 60000, total_steps_16bs - 20000, total_steps_16bs],
        )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=1000 / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )


lr_multiplier_1x = default_X_scheduler(1)

SGD = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)

Adam = L(torch.optim.Adam)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    lr=0.02,
    weight_decay=1e-4,
)

train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=50000,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=5000,
    log_period=20,
    device="cuda"
    # ...
)

constants = dict(
    imagenet_rgb256_mean=[123.675, 116.28, 103.53],
    imagenet_rgb256_std=[58.395, 57.12, 57.375],
    imagenet_bgr256_mean=[103.530, 116.280, 123.675],
    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    # imagenet_bgr256_std=[1.0, 1.0, 1.0],
    imagenet_bgr256_std=[57.375, 57.120, 58.395],
)

model = L(customRCNN)(
    backbone=L(customFPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="FrozenBN",
            ),
            out_features=["res2", "res3", "res4", "res5"],
        ),
        in_features="${.bottom_up.out_features}",
        out_channels=256,
        top_block=L(LastLevelMaxPool)(),
    ),
    proposal_generator=L(customRPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(customROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(customMaskHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    # pixel_mean=constants.imagenet_bgr256_mean,
    # pixel_std=constants.imagenet_bgr256_std,
    pixel_mean=constants["imagenet_bgr256_mean"],
    pixel_std=constants["imagenet_bgr256_std"],
    input_format="BGR",
)