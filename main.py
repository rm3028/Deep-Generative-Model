
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from VAE import *
from AnimeDataset import *
from ExtraDataset import *


if __name__ == '__main__':
    # Initalization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_num = 1000
    batch_size = 128
    learning_rate = 0.0001

    # Read Dataset
    animeDataset = AnimeDataset('data/AnimeDataset')
    dataloader = DataLoader(animeDataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize training
    vae = VAE(96, 1024)
    vae.to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    MSELoss = torch.nn.MSELoss()

    fig, axes = plt.subplots(1, 2)

    # Start training
    for epoch in range(epoch_num):
        for batch_idx, data in enumerate(dataloader):
            image = data['image']
            tags = data['tags']

            image = image.to(device)
            tags = tags.to(device)

            optimizer.zero_grad()
            vae.train()
            output, loss = vae(image)

            loss = MSELoss(output, image) + 0.5 * loss

            loss.backward()
            optimizer.step()

            axes[0].clear()
            axes[1].clear()
            axes[0].imshow(image[0].detach().permute(1, 2, 0).cpu().numpy())
            axes[0].title.set_text('Original Image')
            axes[1].imshow(output[0].detach().permute(1, 2, 0).cpu().numpy())
            axes[1].title.set_text('Generative Image')
            plt.show(block=False)
            plt.pause(0.001)
            #plt.show()

            if batch_idx % 100 == 0 or batch_idx == (dataloader.__len__() - 1):
                trainedNum = (batch_idx + 1) * len(data)
                print(('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}').format(
                    (epoch + 1), trainedNum, len(dataloader.dataset), 100. * trainedNum / len(dataloader.dataset), loss.item()))


    pass
