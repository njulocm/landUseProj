from model import U_Net, AttU_Net, NestedUNet, HRNet
import torch

loss_func = torch.nn.CrossEntropyLoss()
model = HRNet(10)
X = torch.ones((2, 4, 256, 256))
label = torch.zeros(2,256,256)
Y = model(X)
loss = loss_func(Y,label)

a = 0
