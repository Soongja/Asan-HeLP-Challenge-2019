import torch.nn as nn
from .smp import Unet, Linknet, FPN, PSPNet, PAN


weights_path = {
    # 'efficientnet-b0': 'models/pretrained_weights/efficientnet-b0-355c32eb.pth',
    # 'efficientnet-b1': 'models/pretrained_weights/efficientnet-b1-f1951068.pth',
    # 'efficientnet-b2': 'models/pretrained_weights/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'models/pretrained_weights/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'models/pretrained_weights/efficientnet-b4-6ed6700e.pth',
    # 'efficientnet-b5': 'models/pretrained_weights/efficientnet-b5-b6417697.pth',
    # 'efficientnet-b6': 'models/pretrained_weights/efficientnet-b6-c76e70fd.pth',
    'inceptionresnetv2': 'models/pretrained_weights/inceptionresnetv2-520b38e4.pth',
    # 'resnet18': 'models/pretrained_weights/resnet18-5c106cde.pth',
    'resnet34': 'models/pretrained_weights/resnet34-333f7ec4.pth',
    # 'resnet50': 'models/pretrained_weights/resnet50-19c8e357.pth',
    'se_resnext50_32x4d': 'models/pretrained_weights/se_resnext50_32x4d-a260b3a4.pth',
    'se_resnext101_32x4d': 'models/pretrained_weights/se_resnext101_32x4d-3b2fe3d8.pth',
    # 'xception': 'models/pretrained_weights/xception-43020ad28.pth',
    'inceptionv4': 'models/pretrained_weights/inceptionv4-8e4777a0.pth',
    # 'resnext101_32x8d': 'models/pretrained_weights/ig_resnext101_32x8-c38310e5.pth',
}


def get_model(config, pretrained):

    model_architecture = config.MODEL.ARCHITECTURE
    model_encoder = config.MODEL.ENCODER
    in_channels = config.MODEL.IN_CHANNELS

    weights = weights_path[model_encoder] if pretrained else None

    if model_architecture == 'Unet':
        model = Unet(model_encoder, encoder_weights=weights, classes=8,
                     decoder_attention_type='scse', in_channels=in_channels, activation=None)
    elif model_architecture == 'FPN':
        model = FPN(model_encoder, encoder_weights=weights, classes=8, in_channels=in_channels, activation=None)
    elif model_architecture == 'PAN':
        model = PAN(model_encoder, encoder_weights=weights, classes=8, in_channels=in_channels, activation=None)

    print('architecture:', model_architecture, 'encoder:', model_encoder)

    # change activation
    if config.MODEL.CHANGE_ACTIVATION:
        def replace_relu_to_sth(model):
            for child_name, child in model.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(model, child_name, nn.LeakyReLU(inplace=True))
                else:
                    replace_relu_to_sth(child)
        replace_relu_to_sth(model)
        print('relu changed to leaky relu')

    if config.PARALLEL:
        model = nn.DataParallel(model)

    print('[*] num parameters:', count_parameters(model))

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
