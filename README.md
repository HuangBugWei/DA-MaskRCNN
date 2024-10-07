# DA-MaskRCNN
domain adaptation on mask rcnn using detectron2 framework

### train
```bash
python3 customized-train.py --resume --num-gpus 2 --config-file ./configs/dabim-maskrcnn-fpn-config.py
```
flag ```--config-file``` can add with ```da-maskrcnn-fpn-config.py```, ```maskrcnn-fpn-config.py```, ```target-maskrcnn-fpn-config.py```, depends on which experiments.
for transfer training (BIM w/ DA as init. weights for original training process)
```bash
python3 transfer-train.py --num-gpus 2 --config-file ./transfer-maskrcnn-fpn-config.py
```
, and the more default configurations in the experiment can be found in configs. 
### AP evaluation
```bash
CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --config-file ./configs/dabim-maskrcnn-fpn-config.py -i ./ablation-DA-25k
```
### inference
```bash
CUDA_VISIBLE_DEVICES=1 python3 inference.py \
    --model ./0622-gemini/ablation-no-DA-25k/model_0049999.pth \
    --config-file maskrcnn-fpn-config.py \
    --input /home/aicenter/maskrcnn/rebar-target-dataset/ \
    --output ./0622-gemini/ablation-no-DA-25k/0049999 \
    --thres 0.5
```
### t-SNE evaluate domain adaptation performance
```bash
CUDA_VISIBLE_DEVICES=0 python3 tsne.py -i ./ablation-no-DA-25k/ -s noDA --config-file maskrcnn-fpn-config.py -m model_0001999.pth
```
### loss evaluation
```bash
CUDA_VISIBLE_DEVICES=1 python3 evalLoss.py --config-file target-maskrcnn-fpn-config.py -i ./0626-gemini/ablation-vanilla-235
```