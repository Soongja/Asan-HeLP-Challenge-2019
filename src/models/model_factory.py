import torch.nn as nn
from .smp import Unet, Linknet, FPN, PSPNet


weights_path = {
    'resnet18': 'models/pretrained_weights/resnet18-5c106cde.pth'
}


def get_model(config, pretrained):

    model_architecture = config.MODEL.ARCHITECTURE
    model_encoder = config.MODEL.ENCODER
    in_channels = config.MODEL.IN_CHANNELS

    weights = weights_path[model_encoder] if pretrained else None

    if model_architecture == 'Unet':
        model = Unet(model_encoder, encoder_weights=weights, classes=8,
                     decoder_attention_type='scse', in_channels=in_channels)
    # elif model_architecture == 'FPN' or model_architecture == 'PSPNet':
        # model = FPN(model_encoder, encoder_weights=model_pretrained, classes=4)

    print('architecture:', model_architecture, 'encoder:', model_encoder)

    if config.PARALLEL:
        model = nn.DataParallel(model)

    print('[*] num parameters:', count_parameters(model))

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
