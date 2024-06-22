"""Data loader
"""
import logging
import os
from typing import List

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import open3d as o3d

import sys
# add pervious directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from hegn.dataloader.transforms import (
                        Resampler,
                        FixedResampler,
                        RandomJitter,
                        RandomCrop,
                        RandomTransformSE3
                    )



logging.basicConfig(level=logging.INFO)

class ModelNetHdf(Dataset):
    def __init__(self, dataset_path: str, subset: str = 'train', categories: List = None, transform=None):
        """ModelNet40 dataset from PointNet.
        Automatically downloads the dataset if not available

        Args:
            dataset_path (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path

        metadata_fpath = os.path.join(self._root, '{}_files.txt'.format(subset))
        self._logger.info('Loading data from {} for {}'.format(metadata_fpath, subset))

        if not os.path.exists(os.path.join(dataset_path)):
            self._logger.info('Downloading dataset to {}.'.format(dataset_path))
            self._download_dataset(dataset_path)
        else:
            self._logger.info('Dataset found in {}.'.format(dataset_path))

        with open(os.path.join(dataset_path, 'shape_names.txt')) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        with open(os.path.join(dataset_path, '{}_files.txt'.format(subset))) as fid:
            h5_filelist = [line.strip() for line in fid]
            h5_filelist = [x.replace('data/modelnet40_ply_hdf5_2048/', '') for x in h5_filelist]
            h5_filelist = [os.path.join(self._root, f) for f in h5_filelist]

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            self._logger.info('Categories used: {}.'.format(categories_idx))
            self._classes = categories
        else:
            categories_idx = None
            self._logger.info('Using all categories.')

        self._data, self._labels = self._read_h5_files(h5_filelist, categories_idx)
        # self._data, self._labels = self._data[:32], self._labels[:32, ...]
        self._transform = transform
        self._logger.info('Loaded {} {} instances.'.format(self._data.shape[0], subset))

    def __getitem__(self, item):
        sample = {'points': self._data[item, :, :], 'label': self._labels[item], 'idx': np.array(item, dtype=np.int32)}

        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._data.shape[0]

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_h5_files(fnames, categories):

        all_data = []
        all_labels = []

        for fname in fnames:
            f = h5py.File(fname, mode='r')
            data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
            labels = f['label'][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    @staticmethod
    def _download_dataset(dataset_path: str):
        os.makedirs(dataset_path, exist_ok=True)

        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate {}'.format(www))
        os.system('unzip {} -d .'.format(zipfile))
        os.system('mv {} {}'.format(zipfile[:-4], os.path.dirname(dataset_path)))
        os.system('rm {}'.format(zipfile))

    def to_category(self, i):
        return self._idx2category[i]




# to test the dataloader
def main():

    dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048', subset='train', categories=['airplane'])
    print('Classes:', dataset.classes)
    print('Number of samples:', len(dataset))

    sample = dataset[0]
    print('Sample:', sample)
    print('Sample points:', sample['points'].shape)
    print('Sample label:', sample['label'])

    # show the point cloud of the sample with open3d
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(sample['points'][:, :3])


    transform = Compose([
        Resampler(1024),
        RandomJitter(scale=0.01, clip=0.05),
        RandomTransformSE3(rot_mag=0, trans_mag=0.5, scale_range=(2.5, 5.5)),
    ])

    dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048', subset='train', categories=['airplane'],
                          transform=transform)
    sample = dataset[0]
    print('Sample:', sample)
    print('Sample points:', sample['points'].shape)
    print('Sample label:', sample['label'])
    print('Sample idx:', sample['idx'])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample['points'][:, :3])

    o3d.visualization.draw_geometries([pcd, pcd1])
    # print the bounding box of the point cloud


    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['points'].size(), sample_batched['label'])
        if i_batch == 1:
            break


if __name__ == '__main__':
    main()