from model import (
    Model
)

class Client:
    
    def __init__(self, client_id: str, train_data: dict, eval_data: dict, model: type, model_settings: tuple, group: list = None, device: str = "cpu") -> None:
        self._model = model(*model_settings).to(device)
        self.device = device

        self.id = client_id
        self.group = group

        self.train_data = self.model.generate_dataset(train_data)
        self.eval_data = self.model.generate_dataset(eval_data)

    def train(self, num_epochs: int = 1, batch_size: int = 10) -> tuple:
        """Trains on self.model using the client"s train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
        """
        update = self.model.train_model(self.train_data, num_epochs, batch_size, self.device)
        num_train_samples = len(self.train_data)

        return num_train_samples, update

    def test(self, set_to_use: str ="test", batch_size: int = 10) -> dict:
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ["train", "test"].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ["train", "test", "val"]

        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.eval_data

        return self.model.test(data, batch_size, self.device)

    @property
    def num_test_samples(self) -> int:
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data)

    @property
    def num_train_samples(self) -> int:
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data)

    @property
    def num_samples(self) -> int:
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data)

        test_size = 0 
        if self.eval_data is not None:
            test_size = len(self.eval_data)
        return train_size + test_size

    @property
    def model(self) -> Model:
        """Returns this client reference to model being trained"""
        return self._model
