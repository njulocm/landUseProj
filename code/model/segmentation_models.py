import segmentation_models_pytorch as smp
import torch.nn as nn
from model.crfasrnn.crfrnn import CrfRnn
from model.crf import CRFRNN
from utils.crf import DeNormalization


class SmpNet(nn.Module):
    def __init__(self, encoder_name, encoder_weights="imagenet", decoder_attention_type='sesc', in_channels=3,
                 n_class=10, crf_its=5,train_crf_end2end=False):
        super().__init__()
        self.model = smp.UnetPlusPlus(  # UnetPlusPlus
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pretrained weights for encoder initialization
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=n_class,  # model output channels (number of classes in your dataset)
        )
        if not train_crf_end2end:
            for p in self.parameters():
                p.requires_grad = False
        self.crf = CrfRnn(n_class)
        # self.crf = CRFRNN(crf_its, in_channels, n_class)

    def forward(self, x, ori_x, mode="train"):
        pred = self.model(x)
        # x = self.crf(x, pred, mode)
        # x = (x * 255).int().float()
        # ori_x = DeNormalization(mean, std, x, device)
        x = self.crf(ori_x, pred, mode)
        return x
