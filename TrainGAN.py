
from datetime import datetime
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from GAN import *
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
        images = network(noise)

    return images

if __name__ == '__main__':
    # Initalization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_num = 100000
    batch_size = 128
    learning_rate = 0.0002
    image_size = 96
    code_dim = 100

    image_num_g = 25
    cols_num_g = int(math.sqrt(image_num_g))

    # Initalize output folder
    now = datetime.now()
    output_folder = 'results/GAN_' + now.strftime("%y%m%d_%H%M")
    logPath = output_folder + '/log'
    modelPath = output_folder + '/model'

    writer = SummaryWriter(logPath)

    # Read dataset
    animeDataset = AnimeDataset('data/AnimeDataset')
    dataloader = DataLoader(animeDataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize training
    gan = GAN(image_size, code_dim)
    gan.generator.to(device)
    gan.discriminator.to(device)

    optimizer_g = torch.optim.Adam(gan.generator.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(gan.discriminator.parameters(), lr=learning_rate)

    BCELoss = torch.nn.BCELoss(reduction='sum')

    fig_gan, axes_gan = plt.subplots()

    # Start training
    for epoch in range(epoch_num):
        for batch_idx, data in enumerate(dataloader):
            image = data['image']
            image = image.to(device)

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            if torch.rand(1).item() < 0.5:
                gan.generator.train()
                gan.discriminator.eval()

                target_d = torch.ones(image.size(0), 1, device=device)
                predict_g = gan.generator(torch.randn(image.size(0), gan.codeSize, device=device))
                predict_d = gan.discriminator(predict_g)

                loss = BCELoss(predict_d, target_d)
                loss.backward()

                optimizer_g.step()
            else:
                gan.generator.eval()
                gan.discriminator.train()

                target_d = torch.cat((torch.zeros(image.size(0), 1, device=device), torch.ones(image.size(0), 1, device=device)))
                predict_g = gan.generator(torch.randn(image.size(0), gan.codeSize, device=device))
                predict_d = torch.cat((predict_g, image))
                predict_d = gan.discriminator(predict_d)

                loss = BCELoss(predict_d, target_d)
                loss.backward()

                optimizer_d.step()

            predict_g = (predict_g + 1) / 2
            grid = make_grid(predict_g[:(cols_num_g ** 2)], nrow=cols_num_g)

            axes_gan.clear()
            axes_gan.imshow(grid.detach().permute(1, 2, 0).cpu().numpy())
            axes_gan.title.set_text('GAN Generated Images')
            plt.show(block=False)
            plt.pause(0.001)

            if batch_idx % 100 == 0 or batch_idx == (dataloader.__len__() - 1):
                trainedNum = batch_idx * batch_size + len(image)
                print(('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Loss: {:.6f}').format(
                    (epoch + 1), trainedNum, len(dataloader.dataset), 100. * trainedNum / len(dataloader.dataset), loss.item(), loss.item()))

        if epoch > 0 and epoch % 10 == 0:
            modelName = GetModelName()

            SaveModel(gan, modelPath, modelName)
            images = GenerateImages(gan.generator, image_num_g, code_dim)
            images = (images + 1) / 2
            save_image(images, modelPath + '/' + modelName + '.jpg', nrow=cols_num_g)

            writer.add_scalar(logPath + '/Loss', loss.item(), epoch)

            grid = make_grid(images, nrow=cols_num_g)
            writer.add_image('GAN Generated Images', grid, epoch)

    print('Finish!')
