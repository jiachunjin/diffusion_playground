import torch
from torchvision import transforms

transform_dict = {
    'mnist': transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), inplace=True)
    ]),
    # 'OMNIGLOT': transforms.Compose([
    #     transforms.Resize([28, 28]),
    #     transforms.ToTensor(),
    # ]),
    'cifar10': transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]),
}

def data_rescale(X):
    X = 2 * X - 1.0
    return X

def inverse_data_rescale(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)