import os
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import loss
from datasets import get_train_datasets, get_subset_indices
from torch.utils.data import DataLoader, SubsetRandomSampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import provider
from model import HEGN

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')

parser.add_argument('-i', '--dataset_path',
                    default='./data/modelnet40_ply_hdf5_2048',
                    type=str, metavar='PATH',
                    help='path to the processed dataset. Default: ./data/modelnet40_ply_hdf5_2048')

parser.add_argument('--dataset_type', default='modelnet_hdf',
                    choices=['modelnet_hdf', 'bunny', 'armadillo', 'buddha', 'dragon'],
                    metavar='DATASET', help='dataset type (default: modelnet_hdf)')

parser.add_argument('--train_categoryfile', type=str, metavar='PATH', default='./data/modelnet40_half1.txt',
                    help='path to the categories to be trained')  # eg. './dataset/modelnet40_half1.txt'
parser.add_argument('--val_categoryfile', type=str, metavar='PATH', default='./data/modelnet40_half1.txt',
                    help='path to the categories to be val')  # eg. './sampledata/modelnet40_half1.txt'
# settings for input data_loader
parser.add_argument('--test_category_file', type=str, metavar='PATH', default='./data/modelnet40_half2.txt',
                    help='path to the categories to be val')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

MAX_NUM_POINT = 2048


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train_one_epoch(train_loader, optimizer, model, criterion, device):
    # Shuffle train files
    train_chamfer_loss = 0.0
    train_registration_loss = 0.0
    train_loss = 0.0
    start_time = time.time()
    for batch_num, source_points in enumerate(train_loader):
        source_points = source_points['points']

        # Transformations
        R_g = provider.rotate_point_cloud(source_points)
        R_g = torch.tensor(R_g, dtype=torch.float32).to(device)

        t_g = provider.shift_point_cloud(source_points)
        t_g = torch.tensor(t_g, dtype=torch.float32).to(device)

        s_g = provider.scaling_point_cloud(source_points)
        s_g = torch.tensor(s_g, dtype=torch.float32).to(device)

        # Apply transformations
        source_points = source_points.to(device)
        target_points = provider.transform_x_to_y(source_points, R_g, t_g, s_g)

        # plot first point cloud after transformation
        # provider.plot_ply([source_points.detach().cpu()[0], target_points.detach().cpu()[0]], 'input_points')

        # Add jitter noise to both point clouds to make the layers more robust
        source_points = provider.jitter_point_cloud(source_points)
        target_points = provider.jitter_point_cloud(target_points)

        # provider.plot_ply([source_points.cpu().numpy()[0], target_points.cpu().numpy()[0]], 'input_points_Jittered')

        # Swap the dimensions for pytorch
        source_points = source_points.transpose(1, 2).to(device)
        target_points = target_points.transpose(1, 2).to(device)

        # Zero the gradients
        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        R, t, s = model(source_points, target_points)

        # Calculate the loss
        source_points = source_points.permute(0, 2, 1)
        target_points = target_points.permute(0, 2, 1)
        predicted_points = provider.transform_x_to_y(source_points, R, t, s)
        loss, L_R, L_CD = criterion(R, t, s, R_g, t_g, s_g, predicted_points, target_points)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Aggregate validation losses
        train_chamfer_loss += L_CD.item()
        train_registration_loss += L_R.item()
        train_loss += loss.item()
        print(
            f"Batch: {batch_num + 1}, Train mean loss: {train_loss}, Train Chamfer Loss: {train_chamfer_loss}, Train Registration Loss: {train_registration_loss}")

    end_time = time.time()
    fps = len(train_loader.dataset) / (end_time - start_time)
    # Calculate average losses
    avg_train_chamfer_loss = train_chamfer_loss / len(train_loader)
    avg_train_registration_loss = train_registration_loss / len(train_loader)
    avg_loss = train_loss / len(train_loader)

    print(
        f"Train mean loss: {avg_loss}, Train Chamfer Loss: {avg_train_chamfer_loss}, Train Registration Loss: {avg_train_registration_loss}, "
        f"FPS: {fps}")
    return avg_loss, avg_train_chamfer_loss, avg_train_registration_loss, fps


def eval_one_epoch(val_loader, model, criterion, device):
    # Validation loop
    model.eval()
    val_chamfer_loss = 0.0
    val_registration_loss = 0.0
    val_loss = 0.0
    start_time = time.time()
    with torch.no_grad():
        for validation_data in val_loader:
            source_points = validation_data['points']

            # Transformations
            R_g = provider.rotate_point_cloud(source_points)
            R_g = torch.tensor(R_g, dtype=torch.float32).to(device)

            t_g = provider.shift_point_cloud(source_points)
            t_g = torch.tensor(t_g, dtype=torch.float32).to(device)

            s_g = provider.scaling_point_cloud(source_points)
            s_g = torch.tensor(s_g, dtype=torch.float32).to(device)

            # Apply transformations
            source_points = source_points.to(device)
            target_points = provider.transform_x_to_y(source_points, R_g, t_g, s_g)

            # plot first point cloud after transformation
            # provider.plot_ply([source_points[0], target_points[0]], 'input_points')

            # Add jitter noise to both point clouds to make the layers more robust
            source_points = provider.jitter_point_cloud(source_points)
            target_points = provider.jitter_point_cloud(target_points)

            # Swap the dimensions for pytorch
            source_points = source_points.transpose(1, 2)
            target_points = target_points.transpose(1, 2)

            # Forward pass
            R, t, s = model.predict(source_points, target_points)

            # Calculate the loss
            source_points = source_points.permute(0, 2, 1)
            target_points = target_points.permute(0, 2, 1)
            predicted_points = provider.transform_x_to_y(source_points, R, t, s)
            loss, L_R, L_CD = criterion(R, t, s, R_g, t_g, s_g, predicted_points, target_points)

            # Aggregate validation losses
            val_chamfer_loss += L_CD.item()
            val_registration_loss += L_R.item()
            val_loss += loss.item()

    end_time = time.time()
    fps = len(val_loader.dataset) / (end_time - start_time)

    # Calculate average losses
    avg_val_chamfer_loss = val_chamfer_loss / len(val_loader)
    avg_val_registration_loss = val_registration_loss / len(val_loader)
    avg_loss = val_loss / len(val_loader)

    print(
        f"Validation mean loss: {avg_loss}, Validation Chamfer Loss: {avg_val_chamfer_loss}, Validation Registration Loss: {avg_val_registration_loss}, "
        f"FPS: {fps}")

    return avg_loss, avg_val_chamfer_loss, avg_val_registration_loss, fps


if __name__ == '__main__':
    # Log Process information
    log_string('pid: %s' % (str(os.getpid())))
    log_string(str(FLAGS))

    # Get Data
    train_set, val_set = get_train_datasets(FLAGS)

    # take only subset of the data
    train_percentage = 0.1
    test_percentage = 0.1
    

    train_indices = get_subset_indices(train_set, train_percentage)
    test_indices = get_subset_indices(val_set, test_percentage)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_set,
                            batch_size=BATCH_SIZE, shuffle=False, sampler=test_sampler)

    del train_set, val_set

    # check max number of points
    if NUM_POINT > MAX_NUM_POINT:
        log_string("NUM_POINT is greater than MAX_NUM_POINT")
        NUM_POINT = MAX_NUM_POINT
        log_string("NUM_POINT is set to %d" % NUM_POINT)

    # Create summary writers
    train_writer = SummaryWriter(os.path.join(LOG_DIR, 'train'))
    val_writer = SummaryWriter(os.path.join(LOG_DIR, 'val'))
    test_writer = SummaryWriter(os.path.join(LOG_DIR, 'test'))

    # Set the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Configure the optimizer, loss function, and the layers
    # Initialize the layers
    model = HEGN(N=NUM_POINT).to(device)

    criterion = loss.CombinedLoss()

    if OPTIMIZER == 'momentum':
        optimizer = optim.SGD(BASE_LEARNING_RATE, momentum=MOMENTUM)
    else:
        optimizer = optim.Adam(model.parameters(), BASE_LEARNING_RATE, betas=(0.9, 0.99))

    CosineAnnealingLR(optimizer, T_max=MAX_EPOCH)

    # Training Loop
    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_metrics = train_one_epoch(train_loader, optimizer, model, criterion, device)
        validation_metrics = eval_one_epoch(val_loader, model, criterion, device)

        # add Train metrics to tensorboard
        train_writer.add_scalar('Loss', train_metrics[0], epoch)
        train_writer.add_scalar('Chamfer Loss', train_metrics[1], epoch)
        train_writer.add_scalar('Registration Loss', train_metrics[2], epoch)
        train_writer.add_scalar('FPS', train_metrics[3], epoch)

        # add Validation metrics to tensorboard
        val_writer.add_scalar('Loss', validation_metrics[0], epoch)
        val_writer.add_scalar('Chamfer Loss', validation_metrics[1], epoch)
        val_writer.add_scalar('Registration Loss', validation_metrics[2], epoch)
        val_writer.add_scalar('FPS', validation_metrics[3], epoch)

        # Save the variables to disk.
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "layers.ckpt"))
            log_string("Model saved in file: %s" % os.path.join(LOG_DIR, "layers.ckpt"))
