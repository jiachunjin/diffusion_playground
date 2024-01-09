import torch
import torchvision
from .data_transform import transform_dict as trans

class mnist():
    def __init__(self, root, split='train'):
        self.data = torchvision.datasets.MNIST(root, train=(split == 'train'), download=True, transform=trans['mnist'])

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]
    
    def __len__(self):
        return len(self.data)