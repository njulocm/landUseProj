import segmentation_models_pytorch as smp
import torch.nn as nn
from model.crfasrnn.crfrnn import CrfRnn

class SmpNet(nn.Module):
    def __init__(self, encoder_name,encoder_weights="imagenet", in_channels=3,n_class=10):
        super().__init__()
        self.model = smp.UnetPlusPlus(# UnetPlusPlus
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=encoder_weights,     # use `imagenet` pretrained weights for encoder initialization
                in_channels=in_channels,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
        self.crf = CrfRnn(n_class)

    def forward(self, x):
        pred = self.model(x)
        x = self.crf(x, pred)
        return x