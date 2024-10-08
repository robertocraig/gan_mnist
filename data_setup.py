import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse

# Detecta o número de workers
num_workers = os.cpu_count()

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=num_workers, pin_memory=True, persistent_workers=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        # Baixa o dataset MNIST
        datasets.MNIST(root="data", train=True, download=True)
        datasets.MNIST(root="data", train=False, download=True)

    def setup(self, stage=None):
        # Inicialização dos dados para treino e validação
        if stage == 'fit' or stage is None:
            self.mnist_train = datasets.MNIST(root="data", train=True, transform=self.transform)
        if stage == 'test' or stage is None:
            self.mnist_val = datasets.MNIST(root="data", train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

if __name__ == "__main__":
    # Configurando argparse para aceitar argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Carregar o MNIST DataModule")

    # Argumentos opcionais com valores padrão
    parser.add_argument('--batch_size', type=int, default=64, help='Tamanho do batch')
    parser.add_argument('--num_workers', type=int, default=num_workers, help='Número de workers para carregar os dados')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Usar pin_memory')
    parser.add_argument('--persistent_workers', type=bool, default=True, help='Usar persistent_workers')

    # Parsing dos argumentos
    args = parser.parse_args()

    # Instanciando o DataModule com os argumentos passados
    mnist_dm = MNISTDataModule(
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
        persistent_workers=args.persistent_workers
    )
    
    mnist_dm.prepare_data()  # Faz o download do dataset
    mnist_dm.setup('fit')  # Configura o dataloader para o treinamento

    # Exibe número de workers sendo usados
    print(f"Usando {num_workers} workers")

    # Exibe o número de batches no dataloader de treinamento
    train_loader = mnist_dm.train_dataloader()
    print(f"Total de batches no DataLoader de treino: {len(train_loader)}")
    
    # Mostra um exemplo de batch
    for batch in train_loader:
        images, labels = batch
        print(f"Tamanho do batch de imagens: {images.size()}")
        print(f"Tamanho do batch de labels: {labels.size()}")
        break  # Apenas um exemplo, não queremos iterar por todo o dataset aqui        