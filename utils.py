import matplotlib.pyplot as plt
import torch
import argparse
import time
import os

# Objetivo: Esta função salva 10 imagens geradas durante o treinamento em um arquivo PNG. 
# Ela é útil para visualizar as imagens geradas pela GAN ao longo das iterações.
# Como funciona: Ela pega as primeiras 10 imagens do tensor de imagens, 
# converte-as para formato NumPy e as salva em uma grade de 4x5.
# A função também remove os eixos para não atrapalhar a visualização.

def save_10_images(images, iter):
    # Cria a pasta 'training' se ela não existir
    if not os.path.exists('training'):
        os.makedirs('training')
    
    # Cria uma figura com subplots de 2 linhas e 5 colunas (total de 10 imagens)
    fig, axs = plt.subplots(2, 5, figsize=(10, 8))
    
    # Itera sobre as 10 primeiras imagens e seus eixos de subplot correspondentes
    for i, ax in enumerate(axs.flatten()):
        if i < len(images):  # Garante que não vá além do número de imagens
            ax.imshow(images[i].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        ax.axis('off')  # Remove os eixos para melhor visualização das imagens

    # Salva a figura em um arquivo PNG com o nome baseado na iteração
    plt.savefig(f'training/iter_{iter}.png')
    plt.close()  # Fecha a figura para liberar memória

def save_20_images(images, iter):
    # Cria uma figura com subplots de 4 linhas e 5 colunas (total de 20 imagens)
    fig, axs = plt.subplots(4, 5, figsize=(10, 8))
    
    # Itera sobre as imagens e seus eixos de subplot correspondentes
    for i, ax in enumerate(axs.flatten()):
        # Mostra a imagem no eixo correspondente. As imagens são transformadas de tensor para numpy.
        # As imagens são permutadas para que a ordem dos canais esteja correta para matplotlib (C, H, W -> H, W, C)
        ax.imshow(images[i].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        ax.axis('off')  # Remove os eixos para melhor visualização das imagens

    # Salva a figura em um arquivo PNG com o nome baseado na iteração
    plt.savefig(f'training/iter_{iter}.png')
    plt.close()  # Fecha a figura para liberar memória

# Objetivo: Esta função imprime e retorna o tempo total de treinamento em segundos, 
# ajudando a monitorar o desempenho.
# Como funciona: Ela calcula a diferença entre o tempo de início e o tempo de fim, 
# e imprime uma mensagem indicando o tempo total de treinamento.

def print_train_time(start: float, end: float, device: torch.device = None):
    """
    Imprime a diferença entre o tempo de início e fim do treinamento.

    Args:
        start (float): Tempo de início (pode ser obtido usando time.time() ou timeit).
        end (float): Tempo de fim (deve ser obtido quando a computação terminar).
        device (torch.device, optional): Dispositivo no qual o treinamento está ocorrendo. Defaults to None.

    Returns:
        float: Tempo total de treinamento em segundos.
    """
    # Calcula a diferença de tempo
    total_time = end - start

    # Imprime o tempo de treinamento no dispositivo especificado (se houver)
    print(f"Train time on {device}: {total_time:.3f} seconds")

    # Retorna o tempo total para uso posterior
    return total_time

# Objetivo: Esta função calcula a penalidade de gradiente para uma Wasserstein GAN 
# com penalidade de gradiente (WGAN-GP). 
# A penalidade de gradiente é usada para melhorar a estabilidade do treinamento da WGAN, 
# assegurando que os gradientes não explodam ou desapareçam.

# Interpolação entre imagens reais e falsas: A função cria uma interpolação aleatória 
# entre as imagens reais e falsas (geradas pela GAN).

# Cálculo dos gradientes: Passa as imagens interpoladas pelo discriminador
# e calcula os gradientes da saída do modelo em relação a essas imagens interpoladas.

# Penalidade de gradiente: Calcula a norma dos gradientes e aplica a 
# penalidade de gradiente, que força a norma do gradiente a se aproximar de 1.

# Esse termo de penalidade ajuda a manter a norma do gradiente sob controle, 
# melhorando a estabilidade do treinamento.
def _gradient_penalty(model: torch.nn.Module,
                      real_images: torch.tensor,
                      fake_images: torch.tensor,
                      device: torch.device):
    """
    Calcula a penalidade de gradiente para o WGAN GP (Wasserstein GAN com Gradient Penalty).

    Args:
        model (torch.nn.Module): O modelo discriminador.
        real_images (torch.tensor): Um batch de imagens reais.
        fake_images (torch.tensor): Um batch de imagens geradas (falsas).
        device (torch.device): O dispositivo (CPU ou GPU) onde a computação ocorrerá.

    Returns:
        torch.tensor: Penalidade de gradiente calculada.
    """
    # Termo de peso aleatório para interpolação entre dados reais e falsos
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    
    # Interpola entre as imagens reais e falsas
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    # Passa as interpolações pelo discriminador (modelo)
    model_interpolates = model(interpolates)
    
    # Cria um tensor de gradientes fictícios com todas as entradas iguais a 1 para facilitar o cálculo de gradientes
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Calcula os gradientes da saída do modelo em relação aos dados interpolados
    gradients = torch.autograd.grad(
        outputs=model_interpolates,  # Saída do discriminador
        inputs=interpolates,         # Dados interpolados
        grad_outputs=grad_outputs,   # Gradientes fictícios
        create_graph=True,           # Mantém o gráfico para permitir backpropagation
        retain_graph=True,           # Mantém o gráfico para reutilização
        only_inputs=True,            # Calcula gradientes apenas em relação às entradas
    )[0]

    # Ajusta a forma dos gradientes
    gradients = gradients.view(gradients.size(0), -1)

    # Calcula a penalidade de gradiente: (norma do gradiente - 1)²
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    # Retorna a penalidade de gradiente
    return gradient_penalty

# Seção de teste pela linha de comando
if __name__ == "__main__":
    # Configurações de argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Teste as funções do utils.py")
    parser.add_argument('--function', type=str, required=True, help="Função a ser testada: save_10_images, print_train_time ou gradient_penalty")
    parser.add_argument('--iter', type=int, default=0, help="Iteração para salvar as imagens")
    args = parser.parse_args()

    # Testando a função `save_10_images`
    if args.function == 'save_10_images':
        # Criando imagens aleatórias para o teste
        images = torch.randn(10, 1, 28, 28)  # Batch de 10 imagens no formato [B, C, H, W]
        save_10_images(images, args.iter)
        print(f"Imagens salvas na iteração {args.iter}")

    # Testando a função `print_train_time`
    elif args.function == 'print_train_time':
        # Medir o tempo de treino (simulação)
        start_time = time.time()
        time.sleep(2)  # Simula 2 segundos de "treinamento"
        end_time = time.time()
        print_train_time(start_time, end_time, device=torch.device('cpu'))

    # Testando a função `_gradient_penalty`
    elif args.function == 'gradient_penalty':
        # Criando um modelo simples para passar pelas funções
        class SimpleDiscriminator(torch.nn.Module):
            def forward(self, x):
                return torch.sum(x, dim=(1, 2, 3), keepdim=True)  # Simplesmente soma todos os valores

        model = SimpleDiscriminator().to(torch.device('cpu'))
        real_images = torch.randn(4, 1, 28, 28, device=torch.device('cpu'))  # Batch de 4 imagens reais
        fake_images = torch.randn(4, 1, 28, 28, device=torch.device('cpu'))  # Batch de 4 imagens falsas
        penalty = _gradient_penalty(model, real_images, fake_images, device=torch.device('cpu'))
        print(f"Penalidade de gradiente: {penalty.item()}")

    else:
        print("Função inválida. Escolha 'save_10_images', 'print_train_time' ou 'gradient_penalty'.")    
