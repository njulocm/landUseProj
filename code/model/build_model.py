from .FCN import VGGNet, FCN
from .Unet import U_Net


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
    else:
        raise Exception('没有该模型！')
