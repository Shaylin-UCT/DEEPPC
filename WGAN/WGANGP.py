#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training") #Make 100000 once we prove model works
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient") #Might have to play around with this (could be 0 or 0.5 - from official paper)
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--dataset", type=str, default="Elbow", help="Dataset to use [Elbow, Neck_Unlocalized_GAN, Neck_Unlocalized_Self]")
parser.add_argument("--experiment", type=str, default="N/A", help="The experiment number corresponding to notes on tablet")
parser.add_argument("--continueTraining", action='store_true') #If used, it will store true otherwise nothing at all
parser.add_argument("--restartFile", type=str, default=None, help="The path to the .pt file (saved state of the models) we want to continue training.")
opt = parser.parse_args()
print(opt)

current_epoch = 0#to help in retraining
if opt.continueTraining is True:
    print("Continueing Training")
    i = opt.restartFile
    i = i.split("-")
    i = i[-1]
    i = i.split(".")
    i = i[0]
    i = eval(i)
    current_epoch = i
print("_____:", current_epoch)
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
        img = img.view(img.shape[0], *img_shape)
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
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1) 
        #img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

#Some arrays for metrics (mainly graphs)
G_losses = []
D_losses = []

# Loss weight for gradient penalty (gp coefficient)
lambda_gp = 10

# Initialize generator and discriminator
#generator = Generator()
#discriminator = Discriminator()
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

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
#os.makedirs("../../data/mnist", exist_ok=True)
import Data_to_PyDataset
converter = Data_to_PyDataset.DataPrep(opt.dataset, opt.img_size)
data = converter.getData()
dataloader = DataLoader(dataset=data, 
                            batch_size=opt.batch_size, # how many samples per batch? MAKE OPT.BATCHSIZE
                            shuffle=True) # shuffle the data?
#print("dataloader:", dataloader, "of size", len(dataloader))


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
'''
def plotExampleTrainingData(dataloader):
    #Plots training images
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils
    device = torch.device("cuda" if cuda else "cpu")
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8)) #The size of the plot
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:32], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig("GeneratedImages\VanillaGAN\TrainingImagesVanillaGAN.png")
'''

#File for writing output
output = open(f"/home/schetty1/lustre/GeneratedImages/WGANGP/{opt.dataset}/{opt.experiment}/Performance-{opt.dataset}.txt", "w")
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
    data_path = pathlib.Path("/home/schetty1/lustre/GeneratedImages/WGANGP")
    data_path = data_path / ("LossesWGANGP" + opt.dataset + opt.experiment)
    #plt.savefig("GeneratedImages\VanillaGAN\LossesVanillaGAN")
    plt.savefig(data_path)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0
j = 0
for epoch in range(opt.n_epochs):
    epochinternal = j + current_epoch
    j = j+1
    #for i, (imgs, _) in enumerate(dataloader):
    for i, imgs in enumerate(dataloader):
        output = open(f"/home/schetty1/lustre/GeneratedImages/WGANGP/{opt.dataset}/{opt.experiment}/Performance-{opt.dataset}.txt", "a")

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        #print("real_imgs.size:",real_imgs.size())
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epochinternal, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            output.write(
            "%d,%d,%f,%f\n"
            % (epochinternal, i, d_loss.item(), g_loss.item())
        )
            output.write("G_losses")
            output.write(str(G_losses))
            output.write("\n")
            output.write("D_losses")
            output.write(str(D_losses))
            output.write("\n")
            if current_epoch in [200, 201, 202, 198, 199, 203, 204, 205, 206, 197, 196, 195, 207, 208, 209, 210]:
                if batches_done % opt.sample_interval == 0:
                    for count in range(50):
                        save_image(fake_imgs.data[count], f"/home/schetty1/lustre/GeneratedImages/WGANGP/{opt.dataset}/{opt.experiment}/%d{count}-ep{current_epoch}.png"  % batches_done)
            if batches_done % opt.sample_interval == 0:
                for count in range(10):
                    save_image(fake_imgs.data[count], f"/home/schetty1/lustre/GeneratedImages/WGANGP/{opt.dataset}/{opt.experiment}/%d{count}-ep{current_epoch}.png"  % batches_done)
            batches_done += opt.n_critic
        output.close()
    '''
    Save models here
    '''
    #generator._save_to_state_dict("")
    current_epoch = current_epoch + 1
    torch.save({
        "gen_state_dict": generator.state_dict(),
        "dis_state_dict": discriminator.state_dict(),
        "genOptim_state_dict": optimizer_G.state_dict(),
        "disOptim_state_dict": optimizer_D.state_dict(),
    }, f"/home/schetty1/lustre/GeneratedImages/WGANGP/{opt.dataset}/{opt.experiment}/model-{current_epoch}.pt")
visualizeLosses(G_losses=G_losses, D_losses=D_losses)
print("WGANGP Discriminator:", D_losses)
print("WGANGP Generator:", G_losses)
