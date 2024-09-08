import torch
import argparse
import pytorch_lightning as pl
from torch.optim import Adam

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 256, 3, 2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(inplace=True))
        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(inplace=True))
        self.conv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 3, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True))
        self.conv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 1, 4, 2),
            torch.nn.Tanh()
        )
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight, 0.01)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class GeneratorMLP(torch.nn.Module):
    def __init__(self):
        super(GeneratorMLP, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(100, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(inplace=True))
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(inplace=True))
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(1024, 784),
            torch.nn.Tanh()
        )
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, 0.01)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, 100)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)
        return x


class DiscriminatorMLP(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorMLP, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(784, 1024),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(512, 1)
        )
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, 0.01)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 256, 4, 2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(256, 512, 4, 2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(512, 1, 4, 2)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, 0.01)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return x

class GANLightningModule(pl.LightningModule):
    def __init__(self, generator, discriminator, learning_rate=0.0002, beta1=0.5):
        super(GANLightningModule, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.criterion = torch.nn.BCELoss()

        # Salva hiperparâmetros para logging no PyTorch Lightning
        self.save_hyperparameters(ignore=['generator', 'discriminator'])

        # Configura a otimização manual
        self.automatic_optimization = False

    def forward(self, z):
        # O forward é simplesmente o gerador criando uma imagem a partir de um vetor z
        return self.generator(z)

    def configure_optimizers(self):
        # Otimizadores para o gerador e o discriminador
        optimizer_g = Adam(self.generator.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))
        return [optimizer_g, optimizer_d]

    def training_step(self, batch, batch_idx):
        images, _ = batch
        batch_size = images.size(0)
        
        # Otimizadores
        optimizer_g, optimizer_d = self.optimizers()

        # Cria labels reais e falsos para a perda
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # ---------------------------------------------------------------------
        # Treina o discriminador
        # ---------------------------------------------------------------------
        # Imagens reais
        real_preds = self.discriminator(images)
        real_preds = real_preds.view(-1, 1)  # Ajusta a forma de real_preds para (batch_size, 1)
        real_loss = self.criterion(real_preds, real_labels)

        # Imagens falsas (geradas pelo gerador)
        z = torch.randn(batch_size, 100, 1, 1, device=self.device)  # Vetor de ruído
        fake_images = self.generator(z)
        fake_preds = self.discriminator(fake_images.detach())
        fake_preds = fake_preds.view(-1, 1)  # Ajusta a forma de fake_preds para (batch_size, 1)
        fake_loss = self.criterion(fake_preds, fake_labels)

        # Perda total do discriminador
        d_loss = (real_loss + fake_loss) / 2
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Otimiza o discriminador
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()

        # ---------------------------------------------------------------------
        # Treina o gerador
        # ---------------------------------------------------------------------
        # Gera imagens falsas e tenta enganar o discriminador
        fake_images = self.generator(z)  # Reutilizando z
        fake_preds = self.discriminator(fake_images)
        fake_preds = fake_preds.view(-1, 1)  # Ajusta a forma de fake_preds para (batch_size, 1)

        # Gerador quer que o discriminador acredite que as imagens falsas são reais
        g_loss = self.criterion(fake_preds, real_labels)
        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Otimiza o gerador
        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()

        return {'d_loss': d_loss, 'g_loss': g_loss}

# Seção principal para testes e uso via linha de comando
if __name__ == "__main__":
    # Configuração dos argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Testar o Gerador e o Discriminador")
    parser.add_argument('--model', type=str, default='generator', help='Escolha o modelo: generator ou discriminator')
    parser.add_argument('--model_type', type=str, default='conv', help='Escolha o tipo de modelo: conv ou mlp')

    args = parser.parse_args()

    # Inicializando o modelo baseado nos argumentos
    if args.model == 'generator':
        if args.model_type == 'conv':
            model = Generator()
            print("Gerador Conv criado")
            input_tensor = torch.randn((8, 100, 1, 1))  # Aumenta o batch size para 8 para conv generator
        else:
            model = GeneratorMLP()
            print("Gerador MLP criado")
            input_tensor = torch.randn((8, 100))  # Aumenta o batch size para 8 para MLP generator
    elif args.model == 'discriminator':
        if args.model_type == 'conv':
            model = Discriminator()
            print("Discriminador Conv criado")
            input_tensor = torch.randn((8, 1, 28, 28))  # Batch size 8 para conv discriminator
        else:
            model = DiscriminatorMLP()
            print("Discriminador MLP criado")
            input_tensor = torch.randn((8, 1, 28, 28))  # Batch size 8 para MLP discriminator
    else:
        raise ValueError("Modelo inválido. Escolha 'generator' ou 'discriminator'.")

    # Testar o modelo com o tensor de entrada
    output = model(input_tensor)
    print(f"Saída do modelo: {output.shape}")

    # Exibir a quantidade de parâmetros do modelo
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros treináveis: {num_params}")