import torch
import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

import pytorch_lightning as pl

random_seed = 42
torch.manual_seed(random_seed)

BATCH_SIZE = 128
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)

# MNIST DataModule

'''
class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/Users/evelynhoangtran/Universe/MDN projects/gansevelyn/data",
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers


      
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,)),
            ]
        )
        self.dims =  (1,28,28)
        self.num_classes =10


    def prepare_data(self):
        #download
        MNIST(self.data_dir, train= True, download=True)
        MNIST(self.data_dir, train= False, download=True)


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform =self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000,5000])
        
        # Assign test dataset for use in dataloader(s)
        if stage =="test" or stage is None:
            self.mnist_test - MNIST(self.data_dir, train=False, transform = self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

                            # return the data loader with the corresponding dataset
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


'''


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/Users/evelynhoangtran/Universe/MDN projects/gansevelyn/data",
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.dims = (1, 28, 28)
        self.num_classes = 10
        # Important: This property activates manual optimization.

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


# Two models
# Generator


'''
Generator produces fake data and tries 
to trick the Discriminator
'''

# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module): #inherits from nn.module
    def __init__(self, latent_dim): # latent_dim: the number of latent dimensions 
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34] #put to this layer
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28] # put it back again to the shape
    

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x) #linear layer
        x = F.relu(x)   # a relu activation
        x = x.view(-1, 64, 7, 7)  #256 #reshape
        
        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x) # apply the convol layer to update the data in this shape
        x = F.relu(x)  # activation function
        
        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)
        
        # Convolution to 28x28 (1 feature map)
        return self.conv(x)   #put back into this shape



'''
Discriminator inspects the fake fata 
and determines if it's real or fake

- predict if the image is real (from colleected data) 
or fake (generated)

'''

# Discriminator
# Detective: fake or no fake -> 1 output [0,1]

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        #Simple CNN

        # use 2 convolutional 2d layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        
        # a dropout layer
        self.conv2_drop = nn.Dropout2d()
    
        # 2 linear layers at the end 
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,1)    #(input_size, output_size)
    
    # at the end, only use one ouput between 0 ~ 1

    def forward(self, x):
        # apply all all layers, activation functions
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
       
        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 320) # reshape the data
        x = F.relu(self.fc1(x)) #apply the 1st fully connected layer 
        x = F.dropout(x, training=self.training) #dropout layer
        x = self.fc2(x) # fully connected layer
        return torch.sigmoid(x) # this func takes care the ouput x [0,1]



# TODO GAN:
'''
GAN class, this class inherits from pytorch lightning Module
'''

class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002):
        super().__init__()

        self.save_hyperparameters() # make it accessible
        self.generator = Generator(latent_dim = self.hparams.latent_dim)
        self.discriminator = Discriminator()
        # random noise
        self.validation_z = torch.randn(6, self.hparams.latent_dim)

        self.opt_d, self.opt_g = self.configure_optimizers()

        self.automatic_optimization = False


    def forward(self, z):
        return self.generator(z)
    
    # function to define loss
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat,y)
    

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()
        real_imgs, _ = batch

        # sample noise
        print(real_imgs[0])
        print(len(real_imgs[0][0]))
        z = torch.randn(128, self.hparams.latent_dim) # dimensions of image!!!
        z = z.type_as(real_imgs)

        # train generator: maximise the log(D(G(z))) => z = fake images
        if optimizer_idx == 0:
            fake_imgs = self(z) #execute the generator
            y_hat = self.discriminator(fake_imgs)

            y = torch.ones(real_imgs.size(0), 1)
            y = y.type_as(real_imgs)

            g_loss = self.adversarial_loss(y_hat, y)
            self.opt_g.step()

            log_dict = {"g_loss": g_loss}
            return {"loss": g_loss, "progress_bar": log_dict, "log": log_dict}

        # train disctiminator: max log (D(x) + log (1-D(G(z))))
        if optimizer_idx == 1:

            # how well can it label as real
            y_hat_real = self.discriminator(real_imgs)
            y_real = torch.ones(real_imgs.size(0),1)
            y_real = y_real.type_as(real_imgs)
            real_loss = self.adversarial_loss(y_hat_real, y_real)

            # how well can it label as fake
            y_hat_fake = self.discriminator(self(z).detach())
            y_fake = torch.zeros(real_imgs.size(0),1)
            y_fake = y_fake.type_as(real_imgs)

            # log (1-D(G(z)))
            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
            self.opt_d.step()
       
            d_loss = (real_loss + fake_loss) / 2
            log_dict = {"d_loss": d_loss}
            return {"loss": d_loss, "progress_bar": log_dict, "log": log_dict}


    def configure_optimizers(self):
        lr = self.hparams.lr #learning rate
        opt_g = torch.optim.Adam(self.generator.parameters(),lr =lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(),lr =lr)
        return [opt_d, opt_g]
    

    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.lin1.weight) #generate the image 

        sample_imgs = self(z).cpu() # move it into the Cpu
        print('epoch', self.current_epoch)
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.title("Generated Data")
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        plt.show()

    '''
    after each epoch, check how fake the generated images are
    '''
    def on_epoch_end(self):
        self.plot_imgs()



    #----------

    
if __name__ == "__main__":
    dm = MNISTDataModule()
    model = GAN()
    model.plot_imgs()

    '''
  
    if __name__ == "__main__":
        dm = MNISTDataModule()
        model = GAN()
        trainer = pl.Trainer(max_epochs=10)  # You need to set NUM_EPOCHS
        trainer.fit(model, dm)

    '''


