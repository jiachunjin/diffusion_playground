import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import yaml
import datetime

def plot_tensors(X, num_row, num_col, mode):
    """
    Use matplotlib to plot a batch of images stored in a tensor

    Args:
        X: tensor with shape (B, C, H, W)
        num_row: number of rows in the grid
        num_col: number of columns in the grid
        mode: string, 'grey' or 'rgb'        
    """
    assert mode in ['grey', 'rgb']
    N = X.shape[0]
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col * 2, num_row * 2))
    for ax in axes.ravel():
        ax.set_axis_off()
    if mode == 'rgb':
        for i in range(N):
            if num_row == 1:
                axes[i].imshow(T.ToPILImage()(X[i]))
            else:
                axes[i//num_col][i%num_col].imshow(T.ToPILImage()(X[i]))
    elif mode == 'grey':
        X_ = X.squeeze()
        for i in range(N):
            if num_row == 1:
                axes[i].imshow(1 - X_[i, ...], cmap='Greys')
            else:
                axes[i//num_col][i%num_col].imshow(1 - X_[i, ...], cmap='Greys')
    plt.show()
    plt.close(fig)


def save_img_tensors(X, num_row, num_col, mode, save_path):
    assert mode in ['grey', 'rgb']
    N = X.shape[0]
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col * 2, num_row * 2))
    for ax in axes.ravel():
        ax.set_axis_off()
    if mode == 'rgb':
        for i in range(N):
            if num_row == 1:
                axes[i].imshow(T.ToPILImage()(X[i]))
            else:
                axes[i//num_col][i%num_col].imshow(T.ToPILImage()(X[i]))
    elif mode == 'grey':
        X_ = X.squeeze()
        for i in range(N):
            if num_row == 1:
                axes[i].imshow(1 - X_[i, ...], cmap='Greys')
            else:
                axes[i//num_col][i%num_col].imshow(X_[i, ...], cmap='Greys')
    plt.savefig(save_path)
    plt.close(fig)


def dataloader_one_batch(dataset):
    x_list = []
    y_list = []
    for b, (x, y) in enumerate(dataset):
        x_list.append(x)
        y_list.append(y)
    return torch.cat(x_list), torch.cat(y_list)

def create_exp_name(comments=''):
    x = datetime.datetime.now()
    name = x.strftime("%y%m%d") + '-' + x.strftime("%X")
    if len(comments) > 0:
        name = comments + '_' + name
    return name

def load_config(path):
    with open(path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')