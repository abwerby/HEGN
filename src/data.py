import numpy as np
from torch_geometric.datasets import ModelNet
from torch_geometric.data import Dataset
from src.utils.transform import random_rotate, random_scale, random_translate


class CustomModelNetDataset(Dataset):
    """
    Custom dataset class that transforms the ModelNet dataset into a dataset of pairs of point clouds
    with different random transformations.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (i.e., '10', '40').
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed version.
            The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed version.
            The data object will be transformed before being saved to disk. (default: :obj:`None`)
        train (bool, optional): If :obj:`True`, loads the training dataset, otherwise the test dataset. (default: :obj:`True`)
        force_reload (bool, optional): If :obj:`True`, force the data to be re-downloaded/processed, even if it already exists on disk. (default: :obj:`False`)
    """

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        train=True,
        force_reload=False,
    ):
        self.dataset = ModelNet(
            root=root,
            name=name,
            train=train,
            transform=transform,
            pre_transform=pre_transform,
            force_reload=force_reload,
        )

        # Filter out only the 'chair' class for simplicity
        # TODO: Needs to be generalized for any class when pipeline is complete
        self.dataset = self.dataset[self.dataset.y == 8]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): The index of the example.

        Returns:
            list: A list containing the following elements:
                - X_bar (np.ndarray): The source point cloud with random transformations applied.
                - Y_bar (np.ndarray): The target point cloud.
                - Rg (np.ndarray): The rotation matrix that undoes the rotation applied to X_bar.
                - Sg (np.ndarray): The scaling factor that undoes the scaling applied to X_bar.
                - tg (np.ndarray): The translation vector that undoes the translation applied to X_bar.
        """
        Y_bar = self.dataset[idx]
        X_bar = Y_bar.clone()

        # Apply random rotation in the [0, 180] range (degrees) to X_bar
        # The rotation axis to choose is not specified in the paper, so we randomly choose one axis (i.e., x, y, or z)
        X_bar, R = random_rotate(
            X_bar,
            (
                0,
                180,
            ),
            np.random.choice([0, 1, 2]),
        )
        Rg = np.linalg.inv(R)

        # Apply random scaling in the [0.5, 1.5] range to X_bar
        X_bar, S = random_scale(
            X_bar,
            (
                0.5,
                1.5,
            ),
        )
        Sg = 1 / S

        # Apply random translation in the [-0.5, 0.5] range to X_bar
        X_bar, t = random_translate(X_bar, 0.5)
        tg = -t

        # TODO: Jitter both point clouds with Gaussian noise sampled from N(0, 0.01)

        return [X_bar.pos.numpy(), Y_bar.pos.numpy(), Rg, Sg, tg]
