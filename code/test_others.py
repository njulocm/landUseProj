from model import U_Net, AttU_Net, NestedUNet
import torch

model = NestedUNet(4, 10)
X = torch.ones((2, 4, 256, 256))
Y = model(X)

a = 0
