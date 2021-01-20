from model import U_Net
import torch

model = U_Net(4, 10)
X = torch.ones((2, 4, 256, 256))
Y = model(X)

print('end')
