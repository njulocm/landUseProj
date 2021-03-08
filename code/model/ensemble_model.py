import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init


class EnsembleModel(nn.Module):
    def __init__(self, check_point_file_list, device):
        super().__init__()
        self.model_num = len(check_point_file_list)
        self.check_point_file_list = check_point_file_list
        # 加载模型
        self.models = self._load_model(check_point_file_list, device)
        # 设置集成参数
        self.weight = Parameter(torch.Tensor(self.model_num), requires_grad=True)
        init.constant(self.weight, 1.0/self.model_num) # 初始化权重
        # softmax层
        self.softmax_layer=nn.Softmax(dim=1)


    def _load_model(self, check_point_file_list, device):
        models = []
        for ckpt in check_point_file_list:
            model = torch.load(ckpt, map_location=device)
            model.requires_grad_(False)
            models.append(model)
        return models

    def forward(self, x):
        out = None
        for model in self.models:
            temp_out = torch.unsqueeze(model(x), dim=4)
            if out == None:
                out = temp_out
            else:
                out = torch.cat([out, temp_out], dim=4)
        # 加权
        out = torch.matmul(out, self.weight)
        out = self.softmax_layer(out)
        return out
