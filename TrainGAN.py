
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

def GenerateImages(network, image_num, code_dim, code=None):
    if code is not None and torch.is_tensor(code):
        noise = code.to(next(network.parameters()).device)
    else:
        noise = torch.randn((image_num, code_dim), device=next(network.parameters()).device)

    with torch.no_grad():
        network.eval()
        images = network(noise)
        images = (images + 1) / 2

    return images

if __name__ == '__main__':
    # Initalization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_num = 500
    batch_size = 128
    learning_rate = 0.0002
    image_size = 96
    code_dim = 100

    image_num_g = 25
    cols_num_g = int(math.sqrt(image_num_g))
    validation_code = torch.randn((image_num_g, code_dim), device=device)

    # Initalize output folder
    now = datetime.now()
    output_folder = 'results/GAN_' + now.strftime("%y%m%d_%H%M")
    logPath = output_folder + '/log'
    modelPath = output_folder + '/model'

    writer = SummaryWriter(logPath)

    # Read dataset
    animeDataset = AnimeDataset('data/AnimeDataset', activation_type=ActivationType.AT_TANH)
    dataloader = DataLoader(animeDataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize training
    gan = GAN(image_size, code_dim)
    #gan.load_state_dict(torch.load('results/GAN_210128_1053/model/Model_210129_045458.pkl'))
    gan.generator.to(device)
    gan.discriminator.to(device)

    optimizer_g = torch.optim.Adam(gan.generator.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(gan.discriminator.parameters(), lr=learning_rate)

    BCELoss = torch.nn.BCELoss()

    fig_gan, axes_gan = plt.subplots()

    # Start training
    for epoch in range(epoch_num):
        for batch_idx, data in enumerate(dataloader):
            image = data['image']
            image = image.to(device)

            gan.train()

            # Train generator
            optimizer_g.zero_grad()

            target_gd = torch.ones(image.size(0), 1, device=device)
            predict_gg = gan.generator(torch.randn(image.size(0), gan.codeSize, device=device))
            predict_gd = gan.discriminator(predict_gg)

            loss_g = BCELoss(predict_gd, target_gd)
            loss_g.backward()

            optimizer_g.step()

            # Train discriminator
            optimizer_d.zero_grad()

            target_dg = torch.zeros(image.size(0), 1, device=device)
            predict_dg = gan.discriminator(predict_gg.detach())
            target_dr = torch.ones(image.size(0), 1, device=device)
            predict_dr = gan.discriminator(image)

            loss_d = (BCELoss(predict_dg, target_dg) + BCELoss(predict_dr, target_dr)) / 2
            loss_d.backward()

            optimizer_d.step()

            # Show images
            generated_images = ((predict_gg[:(cols_num_g ** 2)]) + 1) / 2
            grid = make_grid(generated_images, nrow=cols_num_g)

            axes_gan.clear()
            axes_gan.imshow(grid.detach().permute(1, 2, 0).cpu().numpy())
            axes_gan.title.set_text('GAN Generated Images')
            plt.show(block=False)
            plt.pause(0.001)

            if batch_idx % 100 == 0 or batch_idx == (dataloader.__len__() - 1):
                trainedNum = batch_idx * batch_size + len(image)
                print(('Train Epoch: {} [{}/{} ({:.0f}%)] Loss Generator: {:.6f} Loss Discriminator: {:.6f}').format(
                    (epoch + 1), trainedNum, len(dataloader.dataset), 100. * trainedNum / len(dataloader.dataset), loss_g.item(), loss_d.item()))


        modelName = GetModelName()

        SaveModel(gan, modelPath, modelName)
        images = GenerateImages(gan.generator, image_num_g, code_dim, validation_code)
        save_image(images, modelPath + '/' + modelName + '.jpg', nrow=cols_num_g)

        writer.add_scalar(logPath + '/Loss_Generator', loss_g.item(), epoch)
        writer.add_scalar(logPath + '/Loss_Discriminator', loss_d.item(), epoch)

        grid = make_grid(images, nrow=cols_num_g)
        writer.add_image('GAN Generated Images', grid, epoch)

    print('Finish!')
