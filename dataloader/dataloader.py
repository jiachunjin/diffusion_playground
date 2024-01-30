from torch.utils.data import DataLoader
from .mnist import mnist
from .toy_2d import sample2d

def prepare_dataloader(root, name, train_batch, test_batch, num_workers, **kwargs):
    if name == 'mnist':
        train_data, test_data = mnist(root, 'train'), mnist(root, 'test')
    elif name == '8gaussians':
        train_data = test_data = sample2d('8gaussians', 10000)
    elif name == '2spirals':
        train_data = test_data = sample2d('2spirals', 10000)
    elif name == 'checkerboard':
        train_data = test_data = sample2d('checkerboard', 10000)

    train_loader = DataLoader(train_data, train_batch, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, shuffle=True, drop_last=False, num_workers=num_workers)

    return train_loader, test_loader