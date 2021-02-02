from model.deeplab.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone,input_channel, output_stride, BatchNorm, pretrained=True):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm,input_channel, pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, input_channel,pretrained)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm,input_channel, pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm,input_channel, pretrained)
    else:
        raise NotImplementedError
