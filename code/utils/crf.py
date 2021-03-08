import torch


def DeNormalization(mean, std, image, device):
    b, c, w, h = image.shape
    t_mean = torch.FloatTensor(mean).view(1, c, 1, 1).expand(b, c, w, h).to(device)
    t_std = torch.FloatTensor(std).view(1, c, 1, 1).expand(b, c, w, h).to(device)

    img = image * t_std + t_mean
    img = (img * 255).int().float()
    return img
