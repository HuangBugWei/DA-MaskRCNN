### method to export model
```bash
python3 export_model.py \
--sample-image /home/aicenter/DA-MaskRCNN/online-steel-image-crop/imgs/34.jpg \
--config-file og-mask_rcnn_fpn.py \
--export-method scripting \
--format torchscript \
--output ./script \
--model /home/aicenter/DA-MaskRCNN/0626-gemini/ablation-DA-25k/model_final.pth \
MODEL.DEVICE cpu
```
