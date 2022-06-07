from torch import (
    nn
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

        self.lenet = nn.Sequential(
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

    def forward(self, x):
        logits = self.lenet(x)
        return logits

    def process_x(self, raw_x_batch):
        return raw_x_batch.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)

    def process_y(self, raw_y_batch):
        return raw_y_batch.long()
