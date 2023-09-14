# 1hash
## 2hash
### 3hash
#### 4hash
##### 5hash
Reference code: [test](./Data_to_PyDataset.py)
# Vanilla GAN
The implementation is derived from [here](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py) implementation. Significant changes include the added ability to continue training.


## Training
### Training a new model
At each epoch, the state of the generator, discrimator and their respective optimizers will be saved to  "model-*epochnumber*.pt" file.

```.bash
# An example run of a VanillaGAN model
python3 [path]/VanillaGAN.py --dataset=Elbow --b2=0.999 --lr=0.002 --experiment=ExperimentName
```
A list of aruements, their usage and default values are provided below. The options marked with a "Yes" are required for each new experiment 

| Config                | Details | Default | ExperimentSpecific
| :-------------------- | :---------- | :---------- | :----------
|`--lr`| Learning rate | 0.0002 | Yes
|`--b1`| b1 hyperparameter of Adam optimizer | 0.5| Yes
|`--b2`|b2 hyperparameter of Adam optimizer | 0.999| Yes
|`--dataset`| The name of the folder from which the model will be trained| Elbow | Yes
|`--experiment`| The name of the experiment. Should correspond to a folder within which files will be saved | N/A | Yes
|`--n_epochs`| The number of training epochs | 400 | -
|`--batch_size`| Batch Size | 64 | -
|`--n_cpu`| Number of CPU threads to use during batch generation| 8| -
|`--latent_dim`| Dimensionality of the latest space| 100| -
|`--img_size`| Dimensions of the square image| 256| -
|`--channels`| Number of image channels | 3| -
|`--sample_interval`| Interval between image samples | 400| -
|`--continueTraining`| If used, training will be restarted | -| -
|`--restartFile`| Path to .pt file from which training should be continued| - | Yes


### Continue training

Upon restarting training, the states of the generator, discrimator and their respective optimizers will be loded from the provided  "model-*epochnumber*.pt" file allowing one to restart training from any epoch. The options above are still required and the training continues as per normal. We recommend moving all files from a run into a seperate folder before restarting training to prevent any file overwriting. 
## Output
The model produces several artefacts all stored in the respective experiment subfolder of [GeneratedImages/VanillaGAN](./GeneratedImages/VanillaGAN) :
**Artefacts**
* Under the "Performance-*dataset+experiment*.txt" file will be generated. It contains a summary of training configurations for debugging as well as training statistics including the Epoch, Batch, Discriminator Loss and Generator Loss. For convience, we print a continuously updated version of the Generator and Discriminator Loss at each update. As such, the last 2 lines of the file contain the Generator Loss and Discriminator loss respectively for the entire training process which is useful for subsequent analysis.
* Images are regularly generated and saved to the folder listed above
* At each epoch, we save the state of both the generator and discriminator as well as their respective optimizers. This is all saved in a "model-*epochnumber*.pt" file.

# WGANGP

The implementation is derived from [here](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py) implementation. Significant changes include the added ability to continue training.


## Training
### Training a new model
At each epoch, the state of the generator, discrimator and their respective optimizers will be saved to  "model-*epochnumber*.pt" file.

```.bash
# An example run of a WGANGP model showing how to restart training
python3 [path]/WGANGP.py --dataset=Elbow --lr=0.00005 --experiment=E2 --continueTraining --restartFile=[path]model-x.pt
```
A list of aruements, their usage and default values are provided below. The options marked with a "Yes" are required for each new experiment 

| Config                | Details | Default | ExperimentSpecific
| :-------------------- | :---------- | :---------- | :----------
|`--lr`| Learning rate | 0.0002 | Yes
|`--b1`| b1 hyperparameter of Adam optimizer | 0.5| Yes
|`--b2`|b2 hyperparameter of Adam optimizer | 0.999| Yes
|`--dataset`| The name of the folder from which the model will be trained| Elbow | Yes
|`--experiment`| The name of the experiment. Should correspond to a folder within which files will be saved | N/A | Yes
|`--n_epochs`| The number of training epochs | 400 | -
|`--batch_size`| Batch Size | 64 | -
|`--n_cpu`| Number of CPU threads to use during batch generation| 8| -
|`--latent_dim`| Dimensionality of the latest space| 100| -
|`--img_size`| Dimensions of the square image| 256| -
|`--channels`| Number of image channels | 3| -
|`--sample_interval`| Interval between image samples | 400| -
|`--continueTraining`| If used, training will be restarted | -| -
|`--restartFile`| Path to .pt file from which training should be continued| - | Yes
|`--clip_value`| Amount to clip the weights by (not used by WGANGP) | 0.01 | -


### Continue training

Upon restarting training, the states of the generator, discrimator and their respective optimizers will be loded from the provided  "model-*epochnumber*.pt" file allowing one to restart training from any epoch. The options above are still required and the training continues as per normal. We recommend moving all files from a run into a seperate folder before restarting training to prevent any file overwriting. 
## Output
The model produces several artefacts all stored in the respective experiment subfolder of [GeneratedImages/WGANGP](./GeneratedImages/WGANGP) :
**Artefacts**
* Under the "Performance-*dataset+experiment*.txt" file will be generated. It contains a summary of training configurations for debugging as well as training statistics including the Epoch, Batch, Discriminator Loss and Generator Loss. For convience, we print a continuously updated version of the Generator and Discriminator Loss at each update. As such, the last 2 lines of the file contain the Generator Loss and Discriminator loss respectively for the entire training process which is useful for subsequent analysis.
* Images are regularly generated and saved to the folder listed above
* At each epoch, we save the state of both the generator and discriminator as well as their respective optimizers. This is all saved in a "model-*epochnumber*.pt" file.

# StyleGAN
For StyleGAN, we use the official StyleGAN2-ADA repo presented [here](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md?plain=1). Note that in our experiments, no ADA is used. For an overview of how to train models, the reader is refered to the aforementioned repo. Some basic operations are:

**Preparing the dataset**\
This will transform each image in the training set into a format usable by StyleGAN. This only needs to be run once. Hereinafter the new dataset will be called **ProcessedDataset**.

```.bash
python3 [path]/stylegan2-ada-pytorch-main/dataset_tool.py --source [path to dataset] --dest [path to output folder] --width 512 --height 512 --resize-filter=box
```
**Training model**\
Initial training:
```.bash
python3 [path]/stylegan2-ada-pytorch-main/train.py --outdir=/[folder to save output] --data=[path to ProcessedDataset] --gpus=1 --aug=noaug --snap=[Save model after "snap" iterations]

```
Resume training:
```.bash
python3 [path]/stylegan2-ada-pytorch-main/train.py --outdir=/[folder to save output] --data=[path to ProcessedDataset] --gpus=1 --aug=noaug --snap=[Save model after "snap" iterations] --resume=[path]/GeneratedImages/StyleGAN2ADA/[subfolder, if needed]/[training run]/[snapshot to resume training from].pkl
```

**Generating Images**\
Note that the seed value is ignored. Instead we inserted a new method at the top of [generate.py](./stylegan2-ada-pytorch-main/generate.py) called randomGenerate(). The **num_numbers** variable should be changed to an appropriate value. This will generate the specified number of random seed values to be used to generate imagery.

```.bash
python3 [path]/stylegan2-ada-pytorch-main/generateOriginal.py  --outdir=/[folder to save output] --trunc=1 --seeds=87,22,269 --network=[path]/GeneratedImages/StyleGAN2ADA/[subfolder, if needed]/[training run]/[snapshot to generate images from].pkl
```


# MedFID
The [baseInception](./MedFID/baseInception.h5) model is the standard InceptionV3 network.

We include two core files: [InceptionTL.py](./MedFID/InceptionTL.py) and [FineTuning.py](./MedFID/FineTuning.py):\
InceptionTL.py transforms InceptionV3 into the model specified in our paper and trains the new layers while keeping the pre-trained (on ImageNet)base model frozen. FineTuning.py also does this but then proceeds to fine-tune the model. The number of epochs of each operation should be specified in the respective scripts.

To generate a MedFID score run the [FIDTests.py](./MedFID/FIDTests.py) script. This requires that the images be split into folders as one MedFID score is provided for the entire folder. Specify the folders in the *folders* array. Ideally, the folders should be subclasses of a common folder, specified as part of the *image_folder* variable. The *src_folder* is the folder of source images. For a fair comparison, the source folder should be a copy of original images but with an equal number of items as the generated imagery being measured. The outputs are written into a CSV file for ease of analysis.

The [FIDConfirmation.py](./MedFID/FIDConfirmation.py) script can be used to generate the augmentations and generate the FID scores to reproduce the validation of MedFID. Note that this requires the model to be fine tuned with the [FineTuning.py](./MedFID/FineTuning.py) script. Alternatively, change the path to the desired trained Inception V3 model in line 15 of [FeatureExtraction.py](./MedFID/FeatureExtraction.py)

## Data_to_PyDataset.py
The [Data_to_PyDataset.py](./Data_to_PyDataset.py) is responsible for converting a folder of images into a dataset usable by PyTorch by extending the built-in Dataset class. The main method is included as a test case. Instantiating the class requires 2 parameters: the image size and the *element*. The element is the class as defined above. It will return an instance of *torch.utils.data.Dataset*. The main method includes a test case that will print "DATALOADER IS CORRECT" if the code is set up correctly as well as an example of an image from the dataset. 

In the GAN, instantiate the class and call *getData()*. The returned object should be passed to the Dataloader within respective *dataset* parameter. 