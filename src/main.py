import argparse
import torch_geometric.transforms as T
from src.data import CustomModelNetDataset
from src.train import train


def main(args):
    transform = T.SamplePoints(args.num_points)
    pre_transform = T.NormalizeScale()
    training_dataset = CustomModelNetDataset(
        root="./data/ModelNet40",
        name="40",
        transform=transform,
        pre_transform=pre_transform,
        train=True,
        force_reload=args.force_reload,
    )
    validation_dataset = CustomModelNetDataset(
        root="./data/ModelNet40",
        name="40",
        transform=transform,
        pre_transform=pre_transform,
        train=False,
        force_reload=args.force_reload,
    )

    train(
        training_dataset,
        validation_dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--force_reload", type=bool, default=False)
    args = parser.parse_args()
    main(args)
