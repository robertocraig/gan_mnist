import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model import Generator, Discriminator, GANLightningModule
from data_setup import MNISTDataModule
import argparse

def main():
    # Configuração do ArgumentParser
    parser = argparse.ArgumentParser(description='Treinamento da GAN no dataset MNIST')

    # Definindo argumentos de linha de comando
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas para treinar (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamanho do batch para o DataLoader (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Taxa de aprendizado para o otimizador (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 para o otimizador Adam (default: 0.5)')

    # Argumentos para acelerador e dispositivos
    if torch.cuda.is_available():
        parser.add_argument('--accelerator', type=str, default='gpu', help='Tipo de acelerador para usar (default: gpu se disponível)')
        parser.add_argument('--devices', type=int, default=1, help='Número de GPUs para usar (default: 1)')
    else:
        parser.add_argument('--accelerator', type=str, default='cpu', help='Tipo de acelerador para usar (default: cpu)')
        parser.add_argument('--devices', type=int, default=1, help='Número de CPUs para usar (default: 1)')

    args = parser.parse_args()

    # Inicializando o datamodule e os modelos do gerador e discriminador
    mnist_dm = MNISTDataModule(batch_size=args.batch_size)
    generator = Generator()
    discriminator = Discriminator()

    # Inicializando o módulo da GAN
    model = GANLightningModule(generator=generator, discriminator=discriminator, learning_rate=args.learning_rate, beta1=args.beta1)

    # Inicializando o trainer com os argumentos atualizados
    trainer = Trainer(
        max_epochs=args.epochs, 
        accelerator=args.accelerator, 
        devices=args.devices
    )

    # Treinamento do modelo
    trainer.fit(model, datamodule=mnist_dm)

    # Salvando o modelo treinado
    trainer.save_checkpoint("gan_mnist.ckpt")
    print("Modelo treinado salvo como gan_mnist.ckpt")

if __name__ == "__main__":
    main()
