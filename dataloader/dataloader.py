from torch.utils.data import DataLoader
from .mnist import mnist

def prepare_dataloader(root, name, train_batch, test_batch, num_workers, **kwargs):
    if name == 'mnist':
        train_data, test_data = mnist(root, 'train'), mnist(root, 'test')
    train_loader = DataLoader(train_data, train_batch, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, test_batch, shuffle=True, drop_last=False, num_workers=num_workers)

    return train_loader, test_loader