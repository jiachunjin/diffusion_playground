from torch.utils.data import DataLoader
from .mnist import mnist
from .cifar10 import cifar10
from .toy_2d import toy_2d

def prepare_dataloader(name, train_batch, test_batch, num_workers, **kwargs):
    if name == 'mnist':
        train_data, test_data = mnist(kwargs['root'], 'train'), mnist(kwargs['root'], 'test')
    elif name == 'cifar10':
        train_data, test_data = cifar10(kwargs['root'], 'train'), mnist(kwargs['root'], 'test')
    elif name == '8gaussians':
        train_data = test_data = toy_2d('8gaussians', kwargs['n_toy_samples'])
    elif name == '2spirals':
        train_data = test_data = toy_2d('2spirals', kwargs['n_toy_samples'])
    elif name == 'checkerboard':
        train_data = test_data = toy_2d('checkerboard', kwargs['n_toy_samples'])

    train_loader = DataLoader(train_data, train_batch, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, shuffle=True, drop_last=False, num_workers=num_workers)

    return train_loader, test_loader