import matplotlib.pyplot as plt
import torchvision.transforms as T


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