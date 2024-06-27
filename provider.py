import os
import sys
import numpy as np
import math
import torch

# These imports used for plotting point clouds
import plotly.graph_objects as go
import plotly.offline as offline
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def rotate_point_cloud(batch_data):
    """Randomly rotate the point clouds to augment the dataset.
       rotation is per shape based along up direction.
       Input:
         BxNx3 array, original batch of point clouds.
       Return:
         Bx3x3 array of rotation matrices for each point cloud in the batch.
    """
    rotation_angles = np.random.uniform(size=batch_data.shape[0]) * np.pi
    cosvals = np.cos(rotation_angles)
    sinvals = np.sin(rotation_angles)
    rotation_matrices = np.array([[[cosval, 0, sinval],
                                   [0, 1, 0],
                                   [-sinval, 0, cosval]] for cosval, sinval in zip(cosvals, sinvals)])
    return rotation_matrices


def scaling_point_cloud(batch_data, scale_low=0.5, scale_high=1.5):
    """Randomly scale the point clouds to augment the dataset.
       scale is per shape.
       Input:
         BxNx3 array, original batch of point clouds.
       Return:
         List of scale factors for each point cloud in the batch.
    """
    scales = np.random.uniform(scale_low, scale_high, batch_data.shape[0])
    return scales


def shift_point_cloud(batch_data, shift_range=0.5):
    """Randomly shift point cloud. Shift is per point cloud.
       Input:
         BxNx3 array, original batch of point clouds.
       Return:
         List of 3-element translation vectors for each point cloud in the batch.
    """
    shifts = np.random.uniform(-shift_range, shift_range, (batch_data.shape[0], 3))
    return shifts


def transform_x_to_y(point_clouds, rotation_matrices, translations, scales):
    """
    Apply transformations to a batch of point clouds.

    Parameters:
    point_clouds (numpy.ndarray): BxNx3 array, original batch of point clouds.
    rotation_matrices (numpy.ndarray): Bx3x3 array of rotation matrices for each point cloud.
    translations (numpy.ndarray): Bx3 array of translation vectors for each point cloud.
    scales (numpy.ndarray): B array of scale factors for each point cloud.

    Returns:
    transformed_clouds (numpy.ndarray): BxNx3 array, transformed batch of point clouds.
    """
    scales = scales.unsqueeze(1).unsqueeze(2)
    # Transpose point clouds to shape B*D*N
    point_clouds = point_clouds.transpose(1, 2)  # Shape B*D*N
    rotated_clouds = torch.bmm(rotation_matrices, point_clouds)
    # Perform element-wise multiplication using broadcasting
    scaled_rotated_clouds = scales * rotated_clouds
    scaled_rotated_clouds = scaled_rotated_clouds.permute(0, 2, 1)
    transformed_clouds = scaled_rotated_clouds + translations[:, None, :]

    return transformed_clouds


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data = torch.tensor(jittered_data, dtype=torch.float32).to(batch_data.device)
    jittered_data += batch_data
    return jittered_data


class ShufflePoints:
    """Shuffles the order of the points"""

    def __call__(self, sample):
        sample['points'] = np.random.permutation(sample['points'])
        return sample


class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        sample['points'] = self._resample(sample['points'], self.num)
        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""

    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


def plot_ply(points_list, title):
    """
    Plot multiple point clouds using plotly.

    Parameters:
    points_list (list of numpy.ndarray): Each element is a Nx3 array representing a point cloud.
    title (str): Title of the plot.
    """
    data = []
    for i, points in enumerate(points_list):
        trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='rgb({},{},{})'.format(i * 50 % 255, i * 100 % 255, i * 150 % 255),
                # Vary color for each point cloud
                opacity=0.8
            )
        )
        data.append(trace)

    fig = go.Figure(data=data)
    fig.update_layout(title=title)
    offline.plot(fig, filename=f"figs/{title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_3dscatter.html",
                 auto_open=False)
