
from datetime import datetime
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from VAE import *
from AnimeDataset import *
from ExtraDataset import *

import os


def GetModelName():
    now = datetime.now()
    timeStr = now.strftime("%y%m%d_%H%M%S")

    return 'Model_' + timeStr

def SaveModel(network, output_folder, modelName):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    torch.save(network.state_dict(), output_folder + '/' + modelName + '.pkl')

def GenerateImages(network, image_num, code_dim):
    noise = torch.randn((image_num, code_dim), device=next(network.parameters()).device)

    with torch.no_grad():
        network.eval()
        images = network.decode(noise)

    return images

if __name__ == '__main__':
    # Initalization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_num = 100
    batch_size = 128
    learning_rate = 0.0001
    image_size = 96
    code_dim = 100

    image_num_g = 25
    cols_num_g = int(math.sqrt(image_num_g))

    # Initalize output folder
    now = datetime.now()
    output_folder = 'results/VAE_' + now.strftime("%y%m%d_%H%M")
    logPath = output_folder + '/log'
    modelPath = output_folder + '/model'

    writer = SummaryWriter(logPath)

    # Read dataset
    animeDataset = AnimeDataset('data/AnimeDataset')
    dataloader = DataLoader(animeDataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize training
    vae = VAE(image_size, code_dim)
    #vae.load_state_dict(torch.load('results/result_210119_1246/model/Model_210119_145259.pkl'))
    vae.to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    MSELoss = torch.nn.MSELoss(reduction='sum')

    fig_og, axes_og = plt.subplots(1, 2)

    # Start training
    for epoch in range(epoch_num):
        for batch_idx, data in enumerate(dataloader):
            image = data['image']
            image = image.to(device)

            optimizer.zero_grad()
            vae.train()
            output, loss_kl = vae(image)

            loss_mse = MSELoss(output, image)
            loss = loss_mse + loss_kl

            loss.backward()
            optimizer.step()

            axes_og[0].clear()
            axes_og[1].clear()
            axes_og[0].imshow(image[0].detach().permute(1, 2, 0).cpu().numpy())
            axes_og[0].title.set_text('Original Image')
            axes_og[1].imshow(output[0].detach().permute(1, 2, 0).cpu().numpy())
            axes_og[1].title.set_text('VAE Generated Image')
            plt.show(block=False)
            plt.pause(0.001)

            if batch_idx % 100 == 0 or batch_idx == (dataloader.__len__() - 1):
                trainedNum = batch_idx * batch_size + len(image)
                print(('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Loss-mse: {:.6f} Loss-kl: {:.6f}').format(
                    (epoch + 1), trainedNum, len(dataloader.dataset), 100. * trainedNum / len(dataloader.dataset), loss.item(), loss_mse.item(), loss_kl.item()))

        modelName = GetModelName()

        SaveModel(vae, modelPath, modelName)
        images = GenerateImages(vae, image_num_g, code_dim)
        save_image(images, modelPath + '/' + modelName + '.jpg', nrow=cols_num_g)

        writer.add_scalar(logPath + '/Loss', loss.item(), epoch)
        writer.add_scalar(logPath + '/Loss-MSE', loss_mse.item(), epoch)
        writer.add_scalar(logPath + '/Loss-KL', loss_kl.item(), epoch)

        grid = make_grid(images, nrow=cols_num_g)
        writer.add_image('VAE Generated Images', grid, epoch)

    print('Finish!')
