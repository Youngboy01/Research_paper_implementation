from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTData:
    def __init__(self, data_dir="./data", batch_size=64):
        self.root = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def get_train_dl(self):
        train_ds = datasets.MNIST(
            root=self.root, train=True, download=True, transform=self.transform
        )
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

    def get_test_dl(self):
        test_ds = datasets.MNIST(
            root=self.root, train=False, download=True, transform=self.transform
        )
        return DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
