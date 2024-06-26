from torch.utils.data import DataLoader
import torch
from src.models.hegn import HEGN, HEGNLoss


def train(training_dataset, validation_dataset, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    model = HEGN(n_knn=20).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.99)
    )
    criterion = HEGNLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in training_loader:
            x_batch, y_batch, Rg, Sg, tg = batch

            x_batch = x_batch.view(-1, 3, 1024)
            y_batch = y_batch.view(-1, 3, 1024)

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            f_X, f_Y = model(x_batch, y_batch)

            # Up to here we extracted node features from the VN-DGCNN model
            # TODO: Implement the rest of the pipeline
            raise NotImplementedError

    return model
