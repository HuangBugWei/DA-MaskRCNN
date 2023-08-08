# python3 customized-train.py --resume --num-gpus 2 --config-file ./da-maskrcnn-fpn-config.py
# sleep 60 
# python3 customized-train.py --resume --num-gpus 2 --config-file ./maskrcnn-fpn-config.py
# sleep 60
# python3 customized-train.py --resume --num-gpus 2 --config-file ./target-maskrcnn-fpn-config.py

# CUDA_VISIBLE_DEVICES=1 python3 tsne.py -i ./ablation-DA-25k/ -s DA --config-file da-maskrcnn-fpn-config.py
# sleep 60
# CUDA_VISIBLE_DEVICES=0 python3 tsne.py -i ./ablation-DA-25k/ -s DA --config-file da-maskrcnn-fpn-config.py --pca
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --config-file da-maskrcnn-fpn-config.py -i ./ablation-DA-25k
# sleep 60
# CUDA_VISIBLE_DEVICES=0 python3 inference.py --model ./0622-gemini/ablation-DA-25k/model_0049999.pth --config-file da-maskrcnn-fpn-config.py --input /home/aicenter/maskrcnn/rebar-target-dataset/ --output ./0622-gemini/ablation-DA-25k/0049999 --thres 0.5
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --model ./0622-gemini/ablation-no-DA-25k/model_0049999.pth --config-file maskrcnn-fpn-config.py --input /home/aicenter/maskrcnn/rebar-target-dataset/ --output ./0622-gemini/ablation-no-DA-25k/0049999 --thres 0.5
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --model ./0622-gemini/ablation-vanilla-235/model_0049999.pth --config-file target-maskrcnn-fpn-config.py --input /home/aicenter/maskrcnn/rebar-target-dataset/ --output ./0622-gemini/ablation-vanilla-235/0049999 --thres 0.5
# sleep 60


# CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --config-file da-maskrcnn-fpn-config.py -i ./ablation-DA-25k --bim
# sleep 60
# CUDA_VISIBLE_DEVICES=0 python3 evaluation.py --config-file maskrcnn-fpn-config.py -i ./ablation-no-DA-25k --bim
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --config-file target-maskrcnn-fpn-config.py -i ./ablation-vanilla-235
# sleep 60
# CUDA_VISIBLE_DEVICES=0 python3 evaluation.py --config-file da-maskrcnn-fpn-config.py -i ./ablation-DA-25k --thres 0.05
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --config-file maskrcnn-fpn-config.py -i ./ablation-no-DA-25k --thres 0.05
# sleep 60
# CUDA_VISIBLE_DEVICES=0 python3 evaluation.py --config-file target-maskrcnn-fpn-config.py -i ./ablation-vanilla-235 --thres 0.05
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 tsne.py -i ./ablation-DA-25k/ -s DA --config-file da-maskrcnn-fpn-config.py -m model_0059999.pth
# sleep 60
# CUDA_VISIBLE_DEVICES=0 python3 tsne.py -i ./ablation-no-DA-25k/ -s noDA --config-file maskrcnn-fpn-config.py -m model_0059999.pth
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 tsne.py -i ./ablation-DA-25k/ -s DA --config-file da-maskrcnn-fpn-config.py -m model_0001999.pth
# sleep 60
# CUDA_VISIBLE_DEVICES=0 python3 tsne.py -i ./ablation-no-DA-25k/ -s noDA --config-file maskrcnn-fpn-config.py -m model_0001999.pth
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --model ./0626-gemini/ablation-DA-25k/model_0059999.pth --config-file da-maskrcnn-fpn-config.py --input /home/aicenter/maskrcnn/rebar-target-dataset/ --output ./0626-gemini/ablation-DA-25k/0059999 --thres 0.5
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --model ./0626-gemini/ablation-no-DA-25k/model_0059999.pth --config-file maskrcnn-fpn-config.py --input /home/aicenter/maskrcnn/rebar-target-dataset/ --output ./0626-gemini/ablation-no-DA-25k/0059999 --thres 0.5
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --model ./0626-gemini/ablation-vanilla-235/model_0059999.pth --config-file target-maskrcnn-fpn-config.py --input /home/aicenter/maskrcnn/rebar-target-dataset/ --output ./0626-gemini/ablation-vanilla-235/0059999 --thres 0.5
# sleep 60
# python3 customized-train.py --resume --num-gpus 2 --config-file ./transfer-maskrcnn-fpn-config.py
# sleep 60
# python3 customized-train.py --resume --num-gpus 2 --config-file ./da-maskrcnn-fpn-config.py
# python3 transfer-train.py --num-gpus 2 --config-file ./transfer-maskrcnn-fpn-config.py
# python3 concate.py

# CUDA_VISIBLE_DEVICES=1 python3 evalLoss.py --config-file target-maskrcnn-fpn-config.py -i ./0626-gemini/ablation-vanilla-235
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 evalLoss.py --config-file maskrcnn-fpn-config.py -i ./0626-gemini/ablation-no-DA-25k --bim
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 evalLoss.py --config-file da-maskrcnn-fpn-config.py -i ./0626-gemini/ablation-DA-25k --bim

# python3 customized-train.py --resume --num-gpus 2 --config-file ./transfer-maskrcnn-fpn-config.py
# sleep 60
# python3 label-num-train.py --resume --num-gpus 2 --config-file ./maskrcnn-fpn-config-10k-random.py
# sleep 60
# CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --config-file transfer-maskrcnn-fpn-config.py -i ./ablation-transfer-0/ --thres 0.5
# sleep 60
# python3 label-num-train.py --resume --num-gpus 2 --config-file ./maskrcnn-fpn-config-500.py
# sleep 60
# python3 label-num-train.py --resume --num-gpus 2 --config-file ./maskrcnn-fpn-config-5k.py
# sleep 60

# python3 label-num-train.py --resume --num-gpus 2 --config-file ./maskrcnn-fpn-config-10k.py
# sleep 60
# python3 label-num-train.py --resume --num-gpus 2 --config-file ./da-maskrcnn-fpn-config-500.py
# sleep 60
# python3 label-num-train.py --resume --num-gpus 2 --config-file ./da-maskrcnn-fpn-config-5k.py
# sleep 60
# python3 label-num-train.py --resume --num-gpus 2 --config-file ./da-maskrcnn-fpn-config-1k.py

CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --config-file da-maskrcnn-fpn-config-500.py -i ./ablation-DA-25k-500/ --thres 0.5
sleep 60
CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --config-file da-maskrcnn-fpn-config-1k.py -i ./ablation-DA-25k-1k/ --thres 0.5
sleep 60
CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --config-file da-maskrcnn-fpn-config-5k.py -i ./ablation-DA-25k-5k/ --thres 0.5