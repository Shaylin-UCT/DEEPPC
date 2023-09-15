# Evaluating Generative Adversarial Networks on Small Medical Datasets: DEEPPC
This codebase contains the code and experiments run as part of the final CS/IT Honours Project at the University of Cape Town under the project titled **Evaluating Generative Adversarial Networks on Small Medical Datasets**. The final paper, authored by Shaylin Chetty and supervised by Geoff Nitschke, can be found [here](./CHTSHA042-DEEPPC-GANs.pdf). The medical dataset used is not available for public use and, as such, is not included in this repo. 

General Notes:
* The trained models for Vanilla GAN and WGANGP are not availble in this repo due to their large size (>3Gb each) however they can be obtained by training the selected GAN as discussed below. 
# Vanilla GAN
The implementation is derived from that of [eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py). Significant changes to the original implementation include the added ability to continue training from a desired epoch. This will load the state of the Generator, Discriminator and thier respective optimizers for the selected epoch. All Vanilla GAN models should be run with the [CondaGAN](./Environments/CondaGAN.yml) environment. The job scripts for each experiment can be found [here](./JobScripts/VanillaGAN/).


## Training
### Training a new model
At each epoch, the state of the generator, discrimator and their respective optimizers will be saved to  "model-*epochnumber*.pt" file.

```.bash
# An example run of a VanillaGAN model
python3 [path]/VanillaGAN.py --dataset=Elbow --b2=0.999 --lr=0.002 --experiment=ExperimentName
```
A list of aruements, their usage and default values are provided below. The options marked with a "Yes" are required for each new experiment.

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


### Continue training

Upon restarting training, the states of the generator, discrimator and their respective optimizers will be loded from the provided  "model-*epochnumber*.pt" file allowing one to restart training from any epoch. The options above are still required along with the following two options. We recommend moving all files from a run into a seperate folder before restarting training to prevent any file overwriting. 
| Config                | Details
| :-------------------- | :---------- 
|`--continueTraining`| If used, training will be restarted from the .pt file specified with the arugment below
|`--restartFile`| Path to .pt file from which training should be continued 
## Output
The model produces several artefacts all stored in the respective experiment subfolder of [GeneratedImages/VanillaGAN](./GeneratedImages/VanillaGAN) :
**Artefacts**
* For each run, a "Performance-*dataset+experiment*.txt" file will be generated. It contains a summary of training configurations for debugging as well as training statistics including the Epoch, Batch, Discriminator Loss and Generator Loss. For convience, we print a continuously updated version of the Generator and Discriminator Loss at each update. As such, the last 2 lines of the file contain the Generator Loss and Discriminator loss respectively for the entire training process which is useful for subsequent analysis.
* Images are regularly generated and saved the folder specified above.
* At each epoch, we save the state of both the generator and discriminator as well as their respective optimizers. This is all saved in a "model-*epochnumber*.pt" file.

# WGANGP
The implementation is derived from that of [eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py). Significant changes to the original implementation include the added ability to continue training from a desired epoch. This will load the state of the Generator, Discriminator and thier respective optimizers for the selected epoch. All WGANGP models should be run with the [CondaGAN](./Environments/CondaGAN.yml) environment. The job scripts for each experiment can be found [here](./JobScripts/WGAN/)

## Training
### Training a new model
At each epoch, the state of the generator, discrimator and their respective optimizers will be saved to  "model-*epochnumber*.pt" file.

```.bash
# An example run of a WGANGP model showing how to restart training
python3 [path]/WGANGP.py --dataset=Elbow --lr=0.00005 --experiment=E2 --continueTraining --restartFile=[path]model-x.pt
```
A list of aruements, their usage and default values are provided below. The options marked with a "Yes" are required for each new experiment.

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
|`--clip_value`| Amount to clip the weights by (not used by WGANGP) | 0.01 | -

### Continue training

Upon restarting training, the states of the generator, discrimator and their respective optimizers will be loded from the provided  "model-*epochnumber*.pt" file allowing one to restart training from any epoch. The options above are still required and the training continues as per normal. We recommend moving all files from a run into a seperate folder before restarting training to prevent any file overwriting. 
| Config                | Details
| :-------------------- | :---------- 
|`--continueTraining`| If used, training will be restarted from the .pt file specified with the arugment below
|`--restartFile`| Path to .pt file from which training should be continued 
## Output
The model produces several artefacts all stored in the respective experiment subfolder of [GeneratedImages/WGANGP](./GeneratedImages/WGANGP) :
**Artefacts**
* For each run, a "Performance-*dataset+experiment*.txt" file will be generated. It contains a summary of training configurations for debugging as well as training statistics including the Epoch, Batch, Discriminator Loss and Generator Loss. For convience, we print a continuously updated version of the Generator and Discriminator Loss at each update. As such, the last 2 lines of the file contain the Generator Loss and Discriminator loss respectively for the entire training process which is useful for subsequent analysis.
* Images are regularly generated and saved the folder specified above.
* At each epoch, we save the state of both the generator and discriminator as well as their respective optimizers. This is all saved in a "model-*epochnumber*.pt" file.

# StyleGAN
For StyleGAN2, we use the official StyleGAN2-ADA repo presented [here](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md?plain=1). Note that in our experiments, no ADA is used. For an overview of how to train models, the reader is refered to the aforementioned repo. Some basic operations are:

**Preparing the dataset**\
This will transform each image in the training set into a format usable by StyleGAN. This only needs to be run once. Hereinafter the new dataset will be called *ProcessedDataset*. Use the [CondaGAN](./Environments/CondaGAN.yml) environment. See the [StyleGANDataprep.job](./JobScripts/StyleGAN/StyleGANDataprep.job) script for an example.

```.bash
python3 [path]/stylegan2-ada-pytorch-main/dataset_tool.py --source [path to dataset] --dest [path to output folder] --width 512 --height 512 --resize-filter=box
```
**Training model**\
Train models use the [CondaGANforStyle](./Environments/CondaGANforStyle.yml) environment.

Initial training (see [StyleGANTrainingNewEnv.job](./JobScripts/StyleGAN/StyleGANTrainingNewEnv.job) for an example):
```.bash
python3 [path]/stylegan2-ada-pytorch-main/train.py --outdir=/[folder to save output] --data=[path to ProcessedDataset] --gpus=1 --aug=noaug --snap=[Save model after "snap" iterations]

```
Resume training (see [StyleGANResTrainingNewEnv.job](./JobScripts/StyleGAN/StyleGANResTrainingNewEnv.job) for an example):
```.bash
python3 [path]/stylegan2-ada-pytorch-main/train.py --outdir=/[folder to save output] --data=[path to ProcessedDataset] --gpus=1 --aug=noaug --snap=[Save model after "snap" iterations] --resume=[path]/GeneratedImages/StyleGAN2ADA/[subfolder, if needed]/[training run]/[snapshot to resume training from].pkl
```

**Generating Images**\
Note that while the seed value needs to be specified, we performed an update that eliminates the need for it in image generation. Instead we inserted a new method at the top of [generate.py](./stylegan2-ada-pytorch-main/generate.py) called randomGenerate(). The **num_numbers** variable should be changed to the number of seed values you wish to generate. The method will then generate the specified number of random seed values. Each seed value will generate a unique image. The *network* is the path to the network snapshot you wish to generate imagery from. For example, to see the images generated at each snapshot, the associated network would have to have images generated seperately. This uses the [GeneratorEnv](./Environments/GeneratorEnv.yml) environment. See [StyleGANGeneratorNewEnv.job](./JobScripts/StyleGAN/StyleGANGeneratorNewEnv.job) for an example.

```.bash
python3 [path]/stylegan2-ada-pytorch-main/generateOriginal.py  --outdir=/[folder to save output] --trunc=1 --seeds=87,22,269 --network=[path]/GeneratedImages/StyleGAN2ADA/[subfolder, if needed]/[training run]/[snapshot to generate images from].pkl
```

# MedFID
The [baseInception](./MedFID/baseInception.h5) model is the standard InceptionV3 network which can be run with [BaseModel.job](./JobScripts/MedFID/BaseModel.job). All MedFID models are run in the [MedFID](./Environments/MedFID.yml) environment. Example job scripts are provided [here](./JobScripts/MedFID/). Note that the pre-trained models used are trained on ImageNet. 

We include two core files: [InceptionTL.py](./MedFID/InceptionTL.py) (run with [InceptionTL.job](./JobScripts/MedFID/InceptionTL.job)) and [FineTuning.py](./MedFID/FineTuning.py) (run with [FineTuning.job](./JobScripts/MedFID/FineTuning.job)):\
InceptionTL.py transforms InceptionV3 into the model specified in our paper and trains the new layers while keeping the pre-trained model frozen. FineTuning.py also does this but then proceeds to fine-tune the model. The number of epochs for each method should be manually specified in the first few lines of the desired .py file as shown in the code sample below. It is the model generated by FineTuning.py that is used in MedFID calculations. 

```python
# ------------------------- Set up training -------------------------                  
steps_per_epoch = 50
epochs = 50
validation_steps = 10
# ------------------------- Set up training -------------------------
```

To generate a MedFID score run the [FIDTests.py](./MedFID/FIDTests.py) script. MedFID scores returns the MedFID score for entire folder of images to gather degree of similarity between all generated images and the original training data. Therefore, if you wish to track the MedFID scores over iterations/epochs, the generated images from each iteration/epoch should be contained within its own folder as a subfolder of the folder called in FIDTests.py. Following the tree structure above, *image_folder* path should be that of the generated images. The values in the *folders* array should be the names of subfolders within *image_folder* that you with to evaluate. For example, when computing the MedFID scores of all Vanilla GAN configurations, the *folders* array will contain the names of folders in each experiment. Under each experiment folder, the generated images (split into folders) will under go the MedFID computation with the outputs are written into a CSV file for ease of analysis. For a fair comparison, we recommend evaluating generated images to a subset of the original training images such that the number of items in each set are equal. 

The [FIDConfirmation.py](./MedFID/FIDConfirmation.py) script can be used to generate the augmentations and generate the FID scores to reproduce the results in the validation section in Appendix A of the paper. Note that this requires the model to be fine tuned with the [FineTuning.py](./MedFID/FineTuning.py) script. Alternatively, change the path to the desired trained Inception V3 model in line 15 of [FeatureExtraction.py](./MedFID/FeatureExtraction.py). The MedFID computations of the augmented validation datasets, as used in the paper, can be done with [FIDTests.py](./MedFID/FIDTests.py). For ease of use, this method has been implemented for the CelebA and Medical datasets in [FIDTestsCeleb.py](./MedFID/FIDTestsCeleb.py) (run with [FIDCeleb.job](./JobScripts/MedFID/FIDCeleb.job)) and [FIDTestsMed.py](./MedFID/FIDTestsMed.py) (run with [FIDMed.job](./JobScripts/MedFID/FIDMed.job)).

## Data_to_PyDataset.py
The [Data_to_PyDataset.py](./Data_to_PyDataset.py) is responsible for converting a folder of images into a dataset usable by PyTorch by extending the built-in Dataset class. The main method is included as a test case. Instantiating the class requires 2 parameters: the image size and the *element*. The element is the class as defined above. It will return an instance of *torch.utils.data.Dataset*. The main method includes a test case that will print "DATALOADER IS CORRECT" if the code is set up correctly as well as an example of an image from the dataset. 

In the GAN, instantiate the class and call *getData()*. The returned object should be passed to the Dataloader within respective *dataset* parameter. 

# Models
## MedFID
For ease of use, the Keras models used to generate MedFID scores are provided. The links to the datasets used for fine tuning the base Inception Network to create FID scores are available in our paper. 
| Model                | Usage 
| :-------------------- | :---------- 
|`--baseInception.h5`| This is the base InceptionV3 network pretrained on ImageNet. This is used by [FineTuning.py](./MedFID/FineTuning.py) and [InceptionTL.py](./MedFID/InceptionTL.py).
|`--FineTunedModel.h5`| The model generated by [FineTuning.py](./MedFID/FineTuning.py)
|`--FineTuned Weights.hdf5`| The weights of the model generated by [FineTuning.py](./MedFID/FineTuning.py). These weights are directly used to compute the FID score in [FeatureExtraction.py](./MedFID/FeatureExtraction.py)
