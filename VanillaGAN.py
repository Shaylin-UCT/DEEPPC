#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate") #Check Values
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient") #Check Values
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient") #Check Values
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") #Depends on Machine
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension") #E.g. a 64x64 image would be 64 -> assumes square images
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--dataset", type=str, default="Elbow", help="Dataset to use [Elbow, Neck_Unlocalized_GAN, Neck_Unlocalized_Self]")
parser.add_argument("--experiment", type=str, default="N/A", help="The experiment number corresponding to notes on tablet")
parser.add_argument("--continueTraining", action='store_true')
parser.add_argument("--restartFile", type=str, default=None, help="The path to the .pt file (saved state of the models) we want to continue training.")
opt = parser.parse_args()
print(opt)

current_epoch = 0
# Method below will be used if we continue training
if opt.continueTraining is True:
    i = opt.restartFile
    i = i.split("-")
    i = i[-1]
    i = i.split(".")
    i = i[0]
    i = eval(i)
    current_epoch = i
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        #print("img", np.shape(img))
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(), #outputs final probability as a sigmoid activation function
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

#Some arrays for metrics (mainly graphs)
G_losses = []
D_losses = []

# Loss function
adversarial_loss = torch.nn.BCELoss() 

# Initialize generator and discriminator=
if opt.continueTraining is False:
    print("Begining training a new model")
    generator = Generator()
    discriminator = Discriminator()
else:
    print(f"Restarting model at {opt.restartFile}")
    generator = Generator()
    discriminator = Discriminator()
    modelToLoad = torch.load(opt.restartFile)
    generator.load_state_dict(modelToLoad["gen_state_dict"])
    discriminator.load_state_dict(modelToLoad["dis_state_dict"])
    generator.train()
    discriminator.train()

# Checks if CUDA is available on the device
print(torch.cuda.is_available())

# If Cuda is not available, the model runs on the CPU. The training time is expected to decrease.
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
import Data_to_PyDataset
converter = Data_to_PyDataset.DataPrep(opt.dataset, opt.img_size)
data = converter.getData()

dataloader = DataLoader(dataset=data, 
                                batch_size=opt.batch_size, # how many samples per batch? MAKE OPT.BATCHSIZE
                                shuffle=True) # shuffle the data?



# Optimizers
if opt.continueTraining is False:
    print("New optimizers")
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
else:
    print(f"Loading optimizer states at {opt.restartFile}")
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    modelToLoad = torch.load(opt.restartFile)
    optimizer_G.load_state_dict(modelToLoad["genOptim_state_dict"])
    optimizer_D.load_state_dict(modelToLoad["disOptim_state_dict"])


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

#File for writing output
output = open(f"/home/schetty1/lustre/GeneratedImages/VanillaGAN/{opt.dataset}/{opt.experiment}/Performance-{opt.dataset}.txt", "w")
output.write(str(opt)+"\n")
output.write("Epoch,Batch,D loss,G loss\n")
output.close()

def visualizeLosses(G_losses, D_losses):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    import pathlib
    data_path = pathlib.Path("/home/schetty1/GANMetrics/VanillaGAN")
    data_path = data_path / ("LossesVanilleGAN" + opt.dataset)
    #plt.savefig("GeneratedImages\VanillaGAN\LossesVanillaGAN")
    plt.savefig(data_path)

j = 0
for epoch in range(opt.n_epochs):
    epochinternal = j + current_epoch
    j = j+1
    for i, imgs in enumerate(dataloader):
        output = open(f"/home/schetty1/lustre/GeneratedImages/VanillaGAN/{opt.dataset}/{opt.experiment}/Performance-{opt.dataset}.txt", "a")
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epochinternal, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        output.write(
            "%d,%d,%f,%f\n"
            % (epochinternal, i, d_loss.item(), g_loss.item())
        )

        #Add stats to lists
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        output.write("G_losses")
        output.write(str(G_losses))
        output.write("\n")
        output.write("D_losses")
        output.write(str(D_losses))
        output.write("\n")
        output.close()
        batches_done = current_epoch * len(dataloader) + i
        if current_epoch in [200, 201, 202, 198, 199, 203, 204, 205, 206]:
            if (batches_done % opt.sample_interval == 0):
                for count in range(50):
                    save_image(gen_imgs.data[count], f"/home/schetty1/lustre/GeneratedImages/VanillaGAN/{opt.dataset}/{opt.experiment}/%d{count}-ep{current_epoch}.png"  % batches_done)
        if (batches_done % opt.sample_interval == 0):
            if current_epoch == 200:
                for count in range(50):
                    print("in for loop")
            else:
                for count in range(10):
                    save_image(gen_imgs.data[count], f"/home/schetty1/lustre/GeneratedImages/VanillaGAN/{opt.dataset}/{opt.experiment}/%d{count}-ep{current_epoch}.png"  % batches_done)
    
    # ---------------------
    #      Save Models
    # ---------------------
    
    current_epoch = current_epoch + 1
    torch.save({
        "gen_state_dict": generator.state_dict(),
        "dis_state_dict": discriminator.state_dict(),
        "genOptim_state_dict": optimizer_G.state_dict(),
        "disOptim_state_dict": optimizer_D.state_dict(),
    }, f"/home/schetty1/lustre/GeneratedImages/VanillaGAN/{opt.dataset}/{opt.experiment}/model-{current_epoch}.pt")

visualizeLosses(G_losses=G_losses, D_losses=D_losses)
