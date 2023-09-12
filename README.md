# 1hash
## 2hash
### 3hash
#### 4hash
##### 5hash
Reference code: [test](./Data_to_PyDataset.py)
# Vanilla GAN
The implementation is derived from [this](#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py) implementation. Significant changes include the added ability to continue training.


## Training
### New model
At each epoch, the state of the generator, discrimator and their respective optimizers will be saved to  "model-*epochnumber*.pt" file.

```.bash
# An example run of a VanillaGAN model
python3 [path]/VanillaGAN.py --dataset=Elbow --b2=0.999 --lr=0.002 --experiment=ExperimentName
```
A list of aruements, their usage and default values are provided below:

| Config                | Details | Default
| :-------------------- | :---------- | :----------
|`--n_epochs`| The number of training epochs | 400
|`--batch_size`| Batch Size | 64
|`--lr`| Learning rate **Change as per experiment specification**| 0.0002 
|`--b1`| b1 hyperparameter of Adam optimizer **Change as per experiment specification** | 0.5
|`--b2`|b2 hyperparameter of Adam optimizer **Change as per experiment specification**| 0.999
|`--n_cpu`| Number of CPU threads to use during batch generation| 8
|`--latent_dim`| Dimensionality of the latest space| 100
|`--img_size`| Dimensions of the square image| 256
|`--channels`| Number of image channels | 3
|`--sample_interval`| Interval between image samples | 400
|`--dataset`||
|`--experiment`| The folder to train from **Change as per experiment specification**| Elbow
|`--continueTraining`| If used, training will be restarted | -
|`--restartFile`| Path to .pt file from which training should be continued **Change as per experiment specification**| - 


### Continue training

Upon restarting training, the states of the generator, discrimator and their respective optimizers will be loded from the provided  "model-*epochnumber*.pt" file allowing one to restart training from any epoch.
## Output
The model produces several artefacts all stored in the respective experiment subfolder of [GeneratedImages/VanillaGAN](./GeneratedImages/VanillaGAN) :
**Artefacts**
* Under the "Performance-*dataset+experiment*.txt" file will be generated. It contains a summary of training configurations for debugging as well as training statistics including the Epoch, Batch, Discriminator Loss and Generator Loss. For convience, we print a continuously updated version of the Generator and Discriminator Loss at each update. As such, the last 2 lines of the file contain the Generator Loss and Discriminator loss respectively for the entire training process which is useful for subsequent analysis.
* Images are regularly generated and saved to the folder listed above
* At each epoch, we save the state of both the generator and discriminator as well as their respective optimizers. This is all saved in a "model-*epochnumber*.pt" file.
# WGANGP



# StyleGAN

# MedFID

## Data_to_PyDataset.py
The [Data_to_PyDataset.py](./Data_to_PyDataset.py) is responsible for converting a folder of images into a dataset usable by PyTorch by extending the built-in Dataset class. The main method is included as a test case. Instantiating the class requires 2 parameters: the image size and the *element*. The element is the class as defined above. It will return an instance of *torch.utils.data.Dataset*. The main method includes a test case that will print "DATALOADER IS CORRECT" if the code is set up correctly as well as an example of an image from the dataset. 

In the GAN, instantiate the class and call *getData()*. The returned object should be passed to the Dataloader within respective *dataset* parameter. 