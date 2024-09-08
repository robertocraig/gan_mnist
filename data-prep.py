import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        # Adiciona normalização para a faixa [-1, 1], comum para GANs
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normaliza para [-1, 1]
        ])

    def prepare_data(self):
        # Baixa o dataset MNIST
        datasets.MNIST(root="data", train=True, download=True)
        datasets.MNIST(root="data", train=False, download=True)

    def setup(self, stage=None):
        # Transforma os dados para treino e validação
        self.mnist_train = datasets.MNIST(root="data", train=True, transform=self.transform)
        self.mnist_val = datasets.MNIST(root="data", train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

if __name__ == "__main__":
    mnist_dm = MNISTDataModule()
    mnist_dm.prepare_data()
    mnist_dm.setup()
    print("Data prepared and setup.")
