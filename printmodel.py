import torch
import torch.nn as nn

#model = torch.load("./ablation-DA-25k-061715/model_final.pth", map_location=torch.device('cpu'$
#for idx, (name, param) in enumerate(model["model"].items()):
    #if idx == 10:
    #    break
#    print(name)
    #print(param)
    # name: str
    # param: Tensor
#print("backbone.bottom_up.res2.0.conv3.norm.weight")
#print(model["model"]["backbone.bottom_up.res2.0.conv3.norm.weight"])

#########
# backbone.fpn_lateral5.weight
#########

modelName = "./0626-gemini/ablation-DA-25k/model_0001999.pth"
modelName = "./0626-gemini/ablation-DA-25k/model_final.pth"
modelName = "./0619-gemini/ablation-no-DA-25k/model_final.pth"
modelName = './ablation-transfer-5/model_0003999.pth'
modelName = './ablation-transfer-5/model_final.pth'
model = torch.load(modelName, map_location=torch.device('cpu'))
# model = torch.load(modelName)
# print("backbone.bottom_up.res2.0.conv3.norm.running_mean")
# print(model["model"]["backbone.bottom_up.res2.0.conv3.norm.running_mean"])
print(model["model"]["backbone.fpn_lateral5.weight"].requires_grad)
# for param in model["model"]["backbone.fpn_lateral5.weight"].parameters():
#     print(param)
