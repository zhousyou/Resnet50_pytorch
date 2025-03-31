import torch
import torch.nn as nn 
import torchvision.models as models
from torchsummary import summary

#resnet50
# model = models.resnet50(weights= False)
# num_features = model.fc.in_features
# model.fc = nn.Linear(in_features=num_features, out_features=100)

#VGG16
# model = models.vgg16(weights=False)

#fastrcnn
model = models.detection.fasterrcnn_resnet50_fpn()

num_features = model.roi_heads.box_predictor.cls_score.in_features

# summary(model, (3, 224,224))
print(model)
print(num_features)
