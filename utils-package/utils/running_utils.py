import torch


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device
