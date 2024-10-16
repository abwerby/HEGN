import torch
from torch.utils.data import DataLoader

import torch.nn as nn
from torchvision.transforms import Compose

import open3d as o3d
import time
import tqdm
import os
import sys
from omegaconf import DictConfig, OmegaConf
import hydra

# add current directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.eval_utils import RMSE, to_h5
from hegn.models.hegn import HEGN, HEGN_Loss
from hegn.dataloader.dataloader import ModelNetHdf
from hegn.dataloader.transforms import (
                        FixedResampler,
                        Resampler,
                        RandomJitter,
                        RandomTransformSE3
                    )

@hydra.main(config_path="../config", config_name="test_config")
def main(cfg: DictConfig):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = cfg.batch_size

    # Create dataset and dataloader
    transform = Compose([
        RandomTransformSE3(rot_mag=cfg.rotational_magnitude,
                           trans_mag=cfg.translational_magnitude,
                            scale_range=cfg.scale_range) if cfg.registration_mode == '9dof' \
        else RandomTransformSE3(rot_mag=cfg.rotational_magnitude,
                                trans_mag=cfg.translational_magnitude,
                                scale_range=None),
        RandomJitter(scale=cfg.jitter_scale, clip=cfg.jitter_clip) if cfg.jitter_scale > 0 else nn.Identity(),
        Resampler(1024),
    ])
                           

    dataset = ModelNetHdf(dataset_path=cfg.dataset_path,
                        subset='test', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # write the dataset to an h5 file to test DeepGMR 
    if cfg.export_h5:
        if cfg.registration_mode == '9dof':
            to_h5(dataset, 'data/test_9dof.h5')
        else:
            to_h5(dataset, 'data/test_6dof.h5')

    # make sure that args are the same as in the training script
    class Args:
        def __init__(self):
            self.device = 'cuda'
            self.vngcnn_in =  [2, 64, 64, 64]
            self.vngcnn_out = [32, 32, 32, 32]
            self.n_knn = [20, 20, 16, 16]
            self.topk = [4, 4, 2, 2]
            self.num_blocks = len(self.vngcnn_in)
    args = Args()

    # Define loss function and optimizer
    criterion = HEGN_Loss()

    # Load the model from the checkpoint
    model = HEGN(args=args).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint_path))
    model.eval()

    # save paths
    if cfg.save_output:
        before_folder = cfg.save_output_dir + '/before'
        after_folder = cfg.save_output_dir + '/after'
        if not os.path.exists(before_folder):
            os.makedirs(before_folder)
        if not os.path.exists(after_folder):
            os.makedirs(after_folder)

    # test loop
    with torch.no_grad():
        running_loss = 0.0
        running_loss_reg = 0.0
        running_loss_chm = 0.0
        RMSE_loss = 0.0
        time_per_batch = []
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            x = batch['points'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
            y = batch['points_ts'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
            t_gt = batch['T'].to(device).to(torch.float32)
            R_gt = batch['R'].to(device).to(torch.float32)
            S_gt = batch['S'].to(device).to(torch.float32)
            transform_gt = batch['transform'].to(device).to(torch.float32)
            curr_batch_size = x.size(0)
            # find centroids of both x and y
            x_centroid = x.mean(dim=2, keepdim=True)
            y_centroid = y.mean(dim=2, keepdim=True)    
            x_par = x - x_centroid
            y_par = y - y_centroid
            # stop the logging
            start_time = time.time()
            R, S = model(x_par, y_par)
            t = y_centroid - torch.matmul(R, x_centroid)
            S = torch.diag_embed(S)
            # uncomment the following line to if you want to test only 6dof
            # S = S_gt
            x_aligned = torch.matmul(R, S @ x) + t
            x_aligned_gt = torch.matmul(R_gt, S_gt @ x) + t_gt.unsqueeze(-1)
            end_time = time.time()
            if cfg.save_output:
                # save the point cloud original and transformed
                pc = o3d.geometry.PointCloud()
                pc1 = o3d.geometry.PointCloud()
                pc1.points = o3d.utility.Vector3dVector(x_aligned_gt[0].cpu().numpy().T)
                pc.points = o3d.utility.Vector3dVector(x_aligned[0].cpu().numpy().T)
                pc1.paint_uniform_color([0, 0, 1])
                pc.paint_uniform_color([1, 0, 0])
                o3d.io.write_point_cloud(f'{after_folder}/pc_{i}.ply', pc1+pc)
                pc.points = o3d.utility.Vector3dVector(x[0].cpu().numpy().T)
                o3d.io.write_point_cloud(f'{before_folder}/pc_{i}.ply', pc1+pc)
            if curr_batch_size == batch_size:
                time_per_batch.append(end_time - start_time)
            loss, loss_reg, loss_chm = criterion(x_aligned, x_aligned_gt, R, S, t, R_gt, S_gt, t_gt)
            # calculate RMSE
            RMSE_loss += RMSE(x_par, R, S, t, R_gt, S_gt, t_gt)
            running_loss += loss.item()
            running_loss_reg += loss_reg.item()
            running_loss_chm += loss_chm.item()


    print(f'Loss: {running_loss/len(dataloader)}')
    print(f'Reg Loss: {running_loss_reg/len(dataloader)}')
    print(f'RMSE: {RMSE_loss/len(dataloader)}')
    print(f'Chm Loss: {running_loss_chm/len(dataloader)}')
    print(f'Average time per batch: {sum(time_per_batch)/len(time_per_batch)}, FPS: {1/(sum(time_per_batch)/len(time_per_batch))}')
    print(f'Max time per batch: {max(time_per_batch)}')
    print(f'Min time per batch: {min(time_per_batch)}')
    print(f'Average time per sample: {sum(time_per_batch)/len(dataloader)/batch_size}, FPS: {1/(sum(time_per_batch)/len(dataloader)/batch_size)}')



if __name__ == '__main__':
    main()