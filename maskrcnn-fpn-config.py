from DAMaskRCNN import model, train, dataloader
from DAMaskRCNN import SGD as optimizer
from DAMaskRCNN import lr_multiplier_1x as lr_multiplier

model.backbone.bottom_up.freeze_at = 2
model.roi_heads.num_classes = 2
train["init_checkpoint"] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train["output_dir"] = "./ablation-no-DA"
model.do_domain = False
