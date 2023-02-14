import torch
import torch.nn as nn
from torchvision import models
import bagnets.pytorchnet

from utils.func import print_msg, select_out_features


def generate_model(cfg):
    out_features = select_out_features(
        cfg.data.num_classes,
        cfg.train.criterion
    )

    model = build_model(
        cfg,
        cfg.train.network,
        out_features,
        cfg.train.pretrained
    )

    if cfg.train.checkpoint:
        weights = torch.load(cfg.train.checkpoint)
        model.load_state_dict(weights, strict=True)
        print_msg('Load weights form {}'.format(cfg.train.checkpoint))

    if cfg.base.device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(cfg.base.device)
    return model


def build_model(cfg, network, num_classes, pretrained=False):
    model = BUILDER[network](pretrained=pretrained)
    if 'resnet' in network or 'resnext' in network or 'shufflenet' in network:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'bagnet' in network:
        if cfg.train.version == 'v1':
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else: # nn.Sequential(*list(model.children())[:-2]) 
            new_model =  list(model.children())[:-2]            
            model = Bagnet_v2(new_model, num_classes)
    elif 'densenet' in network:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'vgg' in network:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif 'mobilenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
    elif 'squeezenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    else:
        raise NotImplementedError('Not implemented network.')

    return model


BUILDER = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'wide_resnet50': models.wide_resnet50_2,
    'wide_resnet101': models.wide_resnet101_2,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    'mobilenet': models.mobilenet_v2,
    'squeezenet': models.squeezenet1_1,
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,
    'bagnet33': bagnets.pytorchnet.bagnet33,
}


class Bagnet_v2(nn.Module):
    def __init__(self, model, num_classes):
        super(Bagnet_v2, self).__init__()
        self.sequential = nn.Sequential(*model)
        self.conv2 = nn.Conv2d(2048, num_classes, kernel_size=(1,1), stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 1), stride=(1,1), padding=0)

    def forward(self, x):
        x = self.sequential(x)
        x = self.conv2(x)        
        n, m = x.shape[2], x.shape[3]
        self.avgpool = nn.AvgPool2d(kernel_size=(n, m), stride=(1,1), padding=0)
        x = self.avgpool(x)
        out = x.view(x.shape[0], -1)
        return out     