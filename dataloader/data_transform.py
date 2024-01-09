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
        transforms.ToTensor()
    ]),
}