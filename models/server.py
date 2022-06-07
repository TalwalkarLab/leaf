import numpy as np
from tqdm import tqdm

import torch

from collections import (
    OrderedDict
)

"""
   TODO: Clients are able to learn but server isn"t learning for some reason.
   Try removing the get/set_params wrappers idk
   or maybe you just came up with something at the gym lmao
"""

class Server:
    
    def __init__(self, model_params: OrderedDict) -> None:
        self.model_params = model_params
        self.selected_clients = []
        self.updates = []

    def select_clients(self, my_round: int, possible_clients: list, num_clients: int = 20) -> list:
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]   

    def train_model(self, num_epochs: int = 1, batch_size: int = 10, clients=None) -> None:
        """Trains self.model_params on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client"s data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        
        for client in tqdm(clients, desc="Training clients", leave=False):
            client.model.load_state_dict(self.model_params)
            num_samples, update = client.train(num_epochs, batch_size)

            self.updates.append((num_samples, update))

    @torch.no_grad()
    def update_model(self) -> None:
        new_model = OrderedDict()
        for param_tensor in self.model_params.keys():
            new_model[param_tensor] = 0

        total_weight = 0
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples

            for param_tensor, layer in client_model.items():
                new_model[param_tensor] += client_samples * layer

        for param_tensor in new_model.keys():
            new_model[param_tensor] /= total_weight

        self.model_params = new_model
        self.updates = []

    def test_model(self, clients_to_test: list, set_to_use: str = "test") -> dict:
        """Tests self.model_params on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ["train", "test"].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in tqdm(clients_to_test, desc=f"Evaluating on {set_to_use} set", leave=False):
            client.model.load_state_dict(self.model_params)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, clients: list = None) -> tuple:
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}

        return ids, groups, num_samples

    def save_model(self, path: str) -> None:
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        torch.save({"model_params": self.model_params}, path)