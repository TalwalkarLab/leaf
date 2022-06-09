"""Interfaces for ClientModel and ServerModel."""

from collections import OrderedDict

import torch
from torch import (
    nn,
    optim
)
from torch.utils.data import (
    Dataset,
    DataLoader
)

from baseline_constants import ACCURACY_KEY

class Model(nn.Module):

    def __init__(self, seed: float, lr: float, optimizer=optim.SGD) -> None:
        super(Model, self).__init__()

        self.lr = lr
        self.seed = seed
        self.optimizer = optimizer

    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 4-tuple consisting of:
                features: A placeholder for the samples" features.
                labels: A placeholder for the samples" labels.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
        """
        return None, None, None, None, None
    
    def generate_dataset(self, data: dict) -> Dataset:
        """
        Generate PyTorch Dataset object from data dicts
        """
        return None

    def train_model(self, data: Dataset, num_epochs: int = 1, batch_size: int = 10, device: str = "cpu") -> OrderedDict:
        """
        Trains the client model.

        Args:
            data: Dict of the form {"x": [list], "y": [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

        self.train()
        for _ in range(num_epochs):
            self.run_epoch(train_dataloader, device)

        update = self.state_dict()
        return update

    def run_epoch(self, dataloader: DataLoader, device: str = "cpu") -> None:

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = self.forward(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def test(self, data: Dataset, batch_size: int = 10, device: str = "cpu") -> dict:
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {"x": [list], "y": [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        size = len(data)
        num_batches = len(test_dataloader)
        self.eval()
        test_loss, correct = 0, 0

        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            pred = self.forward(X)

            test_loss += self.loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        return {ACCURACY_KEY: correct, "loss": test_loss}
