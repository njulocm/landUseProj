from .FCN import VGGNet, FCN
from .Unet import U_Net, AttU_Net, NestedUNet
from .SegNet import SegNet
from .PSPNet import PSPNet
from .deeplab.deeplab import DeepLabV3P
from .HRNet import hrnetv2 as HRNet
from .segmentation_models import SmpUnetpp, SmpDeepLab3p, SmpNet, SmpUnet
from .Unet3p import UNet3Plus_DeepSup
from .ensemble_model import EnsembleModel
import torch


def build_model(model_cfg):
    '''
    根聚输入配置构建模型
    :param model_cfg:
    :return:
    '''

    if model_cfg.type == 'FCN':
        vgg_model = VGGNet(requires_grad=True, show_params=model_cfg.show_params)
        fcn_model = FCN(pretrained_net=vgg_model, n_class=model_cfg.num_classes)
        return fcn_model
    elif model_cfg.type == 'Unet':
        u_net = U_Net(in_ch=model_cfg.input_channel, out_ch=model_cfg.num_classes)
        return u_net
    elif model_cfg.type == 'AttUnet':
        attUnet = AttU_Net(img_ch=model_cfg.input_channel, output_ch=model_cfg.num_classes)
        return attUnet
    elif model_cfg.type == 'Unet3p':
        u_net3p = UNet3Plus_DeepSup(img_ch=model_cfg.input_channel, output_ch=model_cfg.num_classes)
        return u_net3p
    elif model_cfg.type == 'NestedUnet':
        nestedUnet = NestedUNet(in_ch=model_cfg.input_channel, out_ch=model_cfg.num_classes)
        return nestedUnet
    elif model_cfg.type == 'SegNet':
        segNet = SegNet(input_nbr=model_cfg.input_channel, label_nbr=model_cfg.num_classes)
        return segNet
    elif model_cfg.type == 'PSPNet':
        psp_net = PSPNet(layers=model_cfg.layers,
                         in_chans=model_cfg.input_channel,
                         bins=model_cfg.bins,
                         dropout=model_cfg.dropout,
                         classes=model_cfg.num_classes,
                         zoom_factor=model_cfg.zoom_factor,
                         use_ppm=model_cfg.use_ppm,
                         pretrained=model_cfg.pretrained)
        return psp_net
    elif model_cfg.type == 'DeepLabV3P':
        deeplabv3p_model = DeepLabV3P(num_classes=model_cfg.num_classes,
                                      backbone=model_cfg.backbone,
                                      output_stride=model_cfg.out_stride,
                                      sync_bn=model_cfg.sync_bn,
                                      freeze_bn=model_cfg.freeze_bn,
                                      pretrained=model_cfg.pretrained)
        return deeplabv3p_model
    elif model_cfg.type == 'HRNet':
        hrnet = HRNet()
        # hrnet = HRNet(in_ch=model_cfg.input_channel, out_ch=model_cfg.num_classes)
        return hrnet
    elif model_cfg.type == "SmpUnet":
        model = SmpUnet(encoder_name=model_cfg.backbone,
                        encoder_depth=model_cfg.encoder_depth,
                        encoder_weights=model_cfg.encoder_weights,
                        decoder_use_batchnorm=model_cfg.decoder_use_batchnorm,
                        decoder_channels=model_cfg.decoder_channels,
                        in_channels=model_cfg.input_channel,
                        classes=model_cfg.num_classes)
        return model
    elif model_cfg.type == 'SmpUnetpp':
        smpnet = SmpUnetpp(encoder_name=model_cfg.backbone,
                           encoder_weights=model_cfg.encoder_weights,
                           in_channels=model_cfg.input_channel,
                           n_class=model_cfg.num_classes,
                           decoder_attention_type=model_cfg.decoder_attention_type,
                           decoder_channels=model_cfg.decoder_channels)
        return smpnet
    elif model_cfg.type == 'SmpDeepLabV3Plus':
        model = SmpDeepLab3p(
            encoder_name=model_cfg.encoder_name,
            encoder_depth=model_cfg.encoder_depth,
            encoder_weights=model_cfg.encoder_weights,
            encoder_output_stride=model_cfg.encoder_output_stride,
            decoder_channels=model_cfg.decoder_channels,
            decoder_atrous_rates=model_cfg.decoder_atrous_rates,
            in_channels=model_cfg.in_channels,
            classes=model_cfg.classes,
            activation=model_cfg.activation,
            upsampling=model_cfg.upsampling,
            aux_params=model_cfg.aux_params,
        )
        return model
    elif model_cfg.type == 'EnsembleModel':
        model = EnsembleModel(check_point_file_list=model_cfg.check_point_file_list,
                              device=model_cfg.device)
        return model
    elif model_cfg.type == 'CheckPoint':  # 加载已有模型
        model = torch.load(model_cfg.check_point_file, map_location=model_cfg.device)
        print("已加载模型" + model_cfg.check_point_file)
        return model

    else:
        raise Exception('没有该模型！')
