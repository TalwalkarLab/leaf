import random


from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


class Server:
    
    def __init__(self, model):
        self.model = model  # global model of the server.
        self.selected_clients = []
        self.updates = []

    def select_clients(self, possible_clients, num_clients=20):
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
        indices = [i for i, _ in enumerate(possible_clients)]
        chosen_indices = random.sample(indices, num_clients)
        clients_chosen = []
        for i in chosen_indices:
            clients_chosen.append(possible_clients[i])

        self.selected_clients = clients_chosen
        return [(len(c.train_data['y']), len(c.eval_data['y'])) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
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
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        for c in clients:
            self.model.send_to([c])
            sys_metrics[c.id][BYTES_READ_KEY] += self.model.size

            comp, num_samples, update = c.train(num_epochs, batch_size, minibatch)
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((num_samples, update))
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += self.model.size
        return sys_metrics

    def update_model(self):
        self.model.update(self.updates)
        self.updates = []

    def test_model(self, clients_to_test=None):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
        """
        if clients_to_test is None:
            clients_to_test = self.selected_clients
        metrics = {}

        self.model.send_to(clients_to_test)
        
        for client in clients_to_test:
            c_metrics = client.test(self.model.cur_model)
            metrics[client.id] = c_metrics

        return metrics

    def get_clients_test_info(self, clients=None):
        """Returns the ids, hierarchies and num_test_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients
        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_test_samples for c in clients}
        return ids, groups, num_samples
