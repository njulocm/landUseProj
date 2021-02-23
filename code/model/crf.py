import torch.nn as nn
from model.crfasrnn.crfrnn import CrfRnn

class CRF(nn.Module):
    def __init__(self,num_labels, num_iterations=5, crf_init_params=None):
        super(CRF, self).__init__()
        self.crf = CrfRnn(num_labels, num_iterations, crf_init_params)

    def forward(self, image, logit):
        out = self.crf(image, logit)
        return out
