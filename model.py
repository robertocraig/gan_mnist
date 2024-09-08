import torch
import argparse

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
        else:
            model = GeneratorMLP()
            print("Gerador MLP criado")
    elif args.model == 'discriminator':
        if args.model_type == 'conv':
            model = Discriminator()
            print("Discriminador Conv criado")
        else:
            model = DiscriminatorMLP()
            print("Discriminador MLP criado")
    else:
        raise ValueError("Modelo inválido. Escolha 'generator' ou 'discriminator'.")

    # Testar o modelo com um tensor aleatório
    input_tensor = torch.randn((1, 100, 1, 1)) if 'generator' in args.model else torch.randn((1, 1, 28, 28))
    output = model(input_tensor)
    print(f"Saída do modelo: {output.shape}")

    # Exibir a quantidade de parâmetros do modelo
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros treináveis: {num_params}")
