
from loss.CELoss import BCELoss
from models.backbone.resnet import resnet50, resnet34, resnet18, resnet152, resnet101
from models.base_block import LinearClassifier

model_dict = {
    'resnet34': [resnet34(), 512],
    'resnet18': [resnet18(), 512],
    'resnet50': [resnet50(), 2048],
    'resnet101': [resnet101(), 2048],
    'resnet152': [resnet152(), 2048],
}

classifier_dict = {
    'base': LinearClassifier,
}

loss_dict = {
    'bce': BCELoss,
}
