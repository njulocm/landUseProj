import torch
from torch import nn
from torchvision import transforms as T


class EnsembleModel(nn.Module):
    def __init__(self, check_point_file_list, device):
        super().__init__()
        self.model_num = len(check_point_file_list)
        self.check_point_file_list = check_point_file_list
        self.device = device
        # 加载模型
        self.models = self._load_model(check_point_file_list, device)
        self.transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.25]),
        ])

    def _load_model(self, check_point_file_list, device):
        models = []
        for ckpt in check_point_file_list:
            model = torch.load(ckpt, map_location=device)
            model.requires_grad_(False)
            models.append(model)
        return models

    def forward(self, x):
        x = x.to(self.device)
        x = x.permute(0, 3, 1, 2) / 255.0
        x = self.transform(x)

        out = None
        for model in self.models:
            temp_out = torch.nn.functional.softmax(model(x), dim=1)
            if out is None:
                out = temp_out
            else:
                out += temp_out
        out = torch.argmax(out, dim=1) + 1
        return out
