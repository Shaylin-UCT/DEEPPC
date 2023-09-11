'''Assumptions: 
1. Always enter the name of the subfolder in ImagesforResearch'''
import torch
from torchvision import datasets, transforms
from PIL import Image
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        '''folder_path is the path to our images'''
        self.folder_path = folder_path
        self.transform = transform
        self.image_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


class DataPrep():

    def __init__(self, element, img_size):
        #Element = Elbow, Neck, etc. 
        self.element = element
        self.img_size = img_size

    def getDatausingLabels(self):
        #Imports
        import pathlib
        from torchvision import datasets
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        import os

        data_path = pathlib.Path("/home/schetty1/lustre/ImagesforResearch")
        #data_path = pathlib.Path("C:/Users/shayl/OneDrive/Documents/Honors/Research/CodeBase/TrainingData")
        image_path = data_path / self.element 
        print("!!", image_path)
        if image_path.is_dir(): #Check if directory exists
            print("Currently working in:", os.getcwd())
            print(f"{image_path} directory exists.")
            print("Full Path:", os.getcwd() / image_path)
            pass
        else:
            print("Currently working in:", os.getcwd())
            print(f"{image_path} directory DOES NOT exist.")
            print("Full Path:", os.getcwd / image_path)
        data_transform = transforms.Compose(
            [
            # Resize the images to 64x64
            transforms.Resize((self.img_size, self.img_size)),
            # Turn the image into a torch.Tensor
            transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
            #Normalize the tensor
            transforms.Normalize([0.5], [0.5])
            ]
            )
        #Get all images
        image_path_list = list(image_path.glob("*.jpg")) 

        #Create new dataset:
        newData =  datasets.ImageFolder(root=data_path, transform=data_transform,target_transform=None)
        return newData

    def getData(self):
        import torch
        from torchvision import datasets, transforms
        from PIL import Image
        import os
        import pathlib
        data_transform = transforms.Compose(
            [
            # Resize the images to 64x64
            transforms.Resize((self.img_size, self.img_size)),
            # Turn the image into a torch.Tensor
            transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
            #Normalize the tensor
            transforms.Normalize([0.5], [0.5])
            ]
            )
        data_path = pathlib.Path("/home/schetty1/lustre/ImagesforResearch")
        #data_path = pathlib.Path("C:/Users/shayl/OneDrive/Documents/Honors/Research/CodeBase/TrainingData")
        image_path = data_path / self.element
        dataset = CustomDataset(folder_path=image_path, transform=data_transform)
        return dataset

def main():
    folder = input("Enter a folder name:")
    from torch.utils.data import DataLoader
    x = DataPrep(folder, 28)
    dataloader = DataLoader(dataset=x.getData(), 
                                batch_size=64, # how many samples per batch?
                                num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True) # shuffle the data?
    print(dataloader)
    print("len(dataloader):",len(dataloader))
    print("batch size:",dataloader.batch_size)
    #Print number of items in directory to check
    import torch
    cuda = True if torch.cuda.is_available() else False
    import os
    count = 0
    dir_path = f"ImagesforResearch/{folder}"
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    print('File count:', count)
    if ((len(dataloader)-1) * dataloader.batch_size) <= count <= ((len(dataloader)+1) * dataloader.batch_size):
        print("DATALOADER IS CORRECT")
    else:
        print("DATALOADER IS NOT CORRECT")
    print("An example from the dataloader:")
    for i, imgs in enumerate(dataloader):
        print("Batch:", i)
        print("BatchSize:",imgs.size())
        break #Used to only print the first batch

if __name__=="__main__":
    main()
