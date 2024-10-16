"""Data loader
"""
import logging
import os
from typing import List
import pickle
import plyfile
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import open3d as o3d

import sys
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

class ScanObjetDataLoader:
    def __init__(self, data_path=''):
        self.DATA_PATH = data_path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(base_dir)
        sys.path.append(os.path.join(base_dir, 'utils'))

    def save_ply(self, points, filename, colors=None, normals=None):
        vertex = np.core.records.fromarrays(points.transpose(), names='x, y, z', formats='f4, f4, f4')
        n = len(vertex)
        desc = vertex.dtype.descr

        if normals is not None:
            vertex_normal = np.core.records.fromarrays(normals.transpose(), names='nx, ny, nz', formats='f4, f4, f4')
            assert len(vertex_normal) == n
            desc = desc + vertex_normal.dtype.descr

        if colors is not None:
            vertex_color = np.core.records.fromarrays(colors.transpose() * 255, names='red, green, blue',
                                                      formats='u1, u1, u1')
            assert len(vertex_color) == n
            desc = desc + vertex_color.dtype.descr

        vertex_all = np.empty(n, dtype=desc)

        for prop in vertex.dtype.names:
            vertex_all[prop] = vertex[prop]

        if normals is not None:
            for prop in vertex_normal.dtype.names:
                vertex_all[prop] = vertex_normal[prop]

        if colors is not None:
            for prop in vertex_color.dtype.names:
                vertex_all[prop] = vertex_color[prop]

        ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
        ply.write(filename)

    def load_pc_file(self, filename, suncg=False, with_bg=True):
        pc = np.fromfile(os.path.join(self.DATA_PATH, filename), dtype=np.float32)

        if suncg:
            pc = pc[1:].reshape((-1, 3))
        else:
            pc = pc[1:].reshape((-1, 11))

        if with_bg:
            pc = np.array(pc[:, 0:3])
            return pc
        else:
            filtered_idx = np.intersect1d(np.intersect1d(np.where(pc[:, -1] != 0)[0], np.where(pc[:, -1] != 1)[0]),
                                          np.where(pc[:, -1] != 2)[0])
            (values, counts) = np.unique(pc[filtered_idx, -1], return_counts=True)
            max_ind = np.argmax(counts)
            idx = np.where(pc[:, -1] == values[max_ind])[0]
            pc = np.array(pc[idx, 0:3])
            return pc

    def load_data(self, filename, num_points=1024, suncg_pl=False, with_bg_pl=True):
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            print("Data loaded.")

        pcs = []
        labels = []

        print(f"With BG: {with_bg_pl}")
        for entry in data:
            filename = entry["filename"].replace('objects_bin/', '')
            pc = self.load_pc_file(filename, suncg=suncg_pl, with_bg=with_bg_pl)
            label = entry['label']

            if pc.shape[0] < num_points:
                continue

            pcs.append(pc)
            labels.append(label)

        print(f"Number of point clouds: {len(pcs)}")
        print(f"Number of labels: {len(labels)}")

        return pcs, labels

    @staticmethod
    def shuffle_points(pcs):
        for pc in pcs:
            np.random.shuffle(pc)
        return pcs

    @staticmethod
    def get_current_data(pcs, labels, num_points):
        sampled = []
        for pc in pcs:
            if pc.shape[0] < num_points:
                print("Points too few.")
                return None, None
            else:
                idx = np.arange(pc.shape[0])
                np.random.shuffle(idx)
                sampled.append(pc[idx[:num_points], :])

        sampled = np.array(sampled)
        labels = np.array(labels)

        idx = np.arange(len(labels))
        np.random.shuffle(idx)

        return sampled[idx], labels[idx]

    @staticmethod
    def normalize_data(pcs):
        for pc in pcs:
            d = max(np.sum(np.abs(pc)**2, axis=-1)**(1./2))
            pc /= d
        return pcs

    @staticmethod
    def normalize_data_multiview(pcs, num_view=5):
        pcs_norm = []
        for pc_set in pcs:
            pc = []
            for j in range(num_view):
                pc_view = pc_set[j, :, :]
                d = max(np.sum(np.abs(pc_view)**2, axis=-1)**(1./2))
                pc.append(pc_view/d)
            pc = np.array(pc)
            pcs_norm.append(pc)
        pcs_norm = np.array(pcs_norm)
        print("Normalized")
        print(pcs_norm.shape)
        return pcs_norm

    @staticmethod
    def center_data(pcs):
        for pc in pcs:
            centroid = np.mean(pc, axis=0)
            pc -= centroid
        return pcs

    @staticmethod
    def get_current_data_h5(pcs, labels, num_points):
        idx_pts = np.arange(pcs.shape[1])
        np.random.shuffle(idx_pts)

        sampled = pcs[:, idx_pts[:num_points], :]

        idx = np.arange(len(labels))
        np.random.shuffle(idx)

        return sampled[idx], labels[idx]

    @staticmethod
    def get_current_data_withmask_h5(pcs, labels, masks, num_points, shuffle=True):
        idx_pts = np.arange(pcs.shape[1])

        if shuffle:
            np.random.shuffle(idx_pts)

        sampled = pcs[:, idx_pts[:num_points], :]
        sampled_mask = masks[:, idx_pts[:num_points]]

        idx = np.arange(len(labels))

        if shuffle:
            np.random.shuffle(idx)

        return sampled[idx], labels[idx], sampled_mask[idx]

    @staticmethod
    def get_current_data_parts_h5(pcs, labels, parts, num_points):
        idx_pts = np.arange(pcs.shape[1])
        np.random.shuffle(idx_pts)

        sampled = pcs[:, idx_pts[:num_points], :]
        sampled_parts = parts[:, idx_pts[:num_points]]

        idx = np.arange(len(labels))
        np.random.shuffle(idx)

        return sampled[idx], labels[idx], sampled_parts[idx]

    @staticmethod
    def get_current_data_discriminator_h5(pcs, labels, types, num_points):
        idx_pts = np.arange(pcs.shape[1])
        np.random.shuffle(idx_pts)

        sampled = pcs[:, idx_pts[:num_points], :]

        idx = np.arange(len(labels))
        np.random.shuffle(idx)

        return sampled[idx], labels[idx], types[idx]

    @staticmethod
    def load_h5(h5_filename):
        with h5py.File(h5_filename, 'r') as f:
            data = f['data'][:]
            label = f['label'][:]
        return data, label

    @staticmethod
    def load_withmask_h5(h5_filename):
        with h5py.File(h5_filename, 'r') as f:
            data = f['data'][:]
            label = f['label'][:]
            mask = f['mask'][:]
        return data, label, mask

    @staticmethod
    def load_discriminator_h5(h5_filename):
        with h5py.File(h5_filename, 'r') as f:
            data = f['data'][:]
            label = f['label'][:]
            model_type = f['type'][:]
        return data, label, model_type

    @staticmethod
    def load_parts_h5(h5_filename):
        with h5py.File(h5_filename, 'r') as f:
            data = f['data'][:]
            label = f['label'][:]
            parts = f['parts'][:]
        return data, label, parts

    @staticmethod
    def convert_to_binary_mask(masks):
        binary_masks = []
        for mask in masks:
            binary_mask = np.ones(mask.shape)
            bg_idx = np.where(mask == -1)
            binary_mask[bg_idx] = 0
            binary_masks.append(binary_mask)
        return np.array(binary_masks)

    @staticmethod
    def flip_types(types):
        return types == 0


# to test the dataloader
def main():

    ## Test the ModelNetHdf class and the dataloader ##
    # dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048', subset='train', categories=['airplane'])
    # print('Classes:', dataset.classes)
    # print('Number of samples:', len(dataset))

    # sample = dataset[0]
    # print('Sample:', sample)
    # print('Sample points:', sample['points'].shape)
    # print('Sample label:', sample['label'])

    # # show the point cloud of the sample with open3d
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(sample['points'][:, :3])


    # transform = Compose([
    #     Resampler(1024),
    #     RandomJitter(scale=0.01, clip=0.05),
    #     RandomTransformSE3(rot_mag=0, trans_mag=0.5, scale_range=(2.5, 5.5)),
    # ])

    # dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048', subset='train', categories=['airplane'],
    #                       transform=transform)
    # sample = dataset[0]
    # print('Sample:', sample)
    # print('Sample points:', sample['points'].shape)
    # print('Sample label:', sample['label'])
    # print('Sample idx:', sample['idx'])

    # pcd = o3d.geometry.PointCloud()
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(sample['points'][:, :3])
    # pcd.points = o3d.utility.Vector3dVector(sample['points_ts'][:, :3])

    # o3d.visualization.draw_geometries([pcd, pcd1])
    # # print the bounding box of the point cloud


    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['points'].size(), sample_batched['label'])
    #     if i_batch == 1:
    #         break
    
    ## Test the PointCloudDataLoader class ##
    loader = ScanObjetDataLoader()
    
    h5_file = '/export/home/werbya/thesis/HEGN/data/scanobjectnn/main_split/test_objectdataset_augmented25_norot.h5'
    h5_data, h5_labels = loader.load_h5(h5_file)

    print(f"Loaded H5 data shape: {h5_data.shape}")
    print(f"Loaded H5 labels shape: {h5_labels.shape}")

    # Get a batch of data from H5 file
    num_points = 1024
    h5_batch_pcs, h5_batch_labels = loader.get_current_data_h5(h5_data, h5_labels, num_points)

    print(f"H5 batch shape: {h5_batch_pcs.shape}")
    print(f"H5 labels shape: {h5_batch_labels.shape}")

    binary_masks = loader.convert_to_binary_mask(np.random.randint(-1, 2, size=(10, 1024)))
    print(f"Binary masks shape: {binary_masks.shape}")
    
    # show the point cloud of the sample with open3d
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(h5_batch_pcs[0, :, :])
    o3d.visualization.draw_geometries([pcd1])


if __name__ == '__main__':
    main()