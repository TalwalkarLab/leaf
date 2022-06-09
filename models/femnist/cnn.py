from torch import (
    nn,
    Tensor
)
from torch.utils.data import (
    TensorDataset
)

from model import (
    Model
)


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        super(ClientModel, self).__init__(seed, lr)

        self.num_classes = num_classes
        
        conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding="same")
        pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same")
        pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        fc1 = nn.Linear(7 * 7 * 64, 2048)
        fc2 = nn.Linear(2048, self.num_classes)

        self.cnn = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool1,
            conv2,
            nn.ReLU(),
            pool2,
            nn.Flatten(),
            fc1,
            nn.ReLU(),
            fc2
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)
    
    def generate_dataset(self, data: dict) -> TensorDataset:
        return TensorDataset(
            Tensor(data["x"]).reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE),
            Tensor(data["y"]).long()
        )

    def forward(self, x):
        logits = self.cnn(x)

        return logits
