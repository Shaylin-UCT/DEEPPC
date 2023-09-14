import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import os
import numpy as np
import csv
import FeatureExtraction

# ----------------------------------------- Info -----------------------------------------

# Take image path -> prepareImage (generates tensor) -> selected method -> displayImage (shows the image to the screen, needs the type of trans. as string)


def prepareImage(image_path):
# Load and process image to generate a tensor
    image = Image.open(image_path).convert('RGB')

    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Apply preprocessing
    image_tensor = preprocess(image)
    return image_tensor

def displayImage(image_tensor, originalImagePath,transformation, parameters):
    if transformation in ["swirl", "saltandpepper"]:
        image_tensor = transforms.ToTensor()(image_tensor)
    # Convert back to [0, 255] range for visualization
    image_tensor = (image_tensor * 255.0).clamp(0, 255).to(torch.uint8)

    # Convert tensor to numpy array
    image_tensor_np = image_tensor.permute(1, 2, 0).numpy()

    # Display the noisy image using matplotlib
    plt.imshow(image_tensor_np)
    plt.axis('off')
    #plt.show()
    output_path = f"ImagesforResearch\\FIDConfirmationImages\\{transformation}\\{parameters}"
    if os.path.exists(output_path):
        pass
    else:
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, originalImagePath), bbox_inches='tight', pad_inches = 0.0)
    #Save image

#Block argumentation
def cover_with_random_boxes(image_path, originalImagePath):
    print(image_path)
    num_boxes = 100
    max_box_area = 50
    areas = [50, 100,500,1000, 5000,10000, 50000, 100000] #Number of blocks to place on image
    for max_box_area in areas:
        img = Image.open(image_path)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        try:
            for _ in range(num_boxes):
                box_width = random.randint(1, int(max_box_area ** 0.5))
                box_height = random.randint(1, int(max_box_area ** 0.5))
                x1 = random.randint(0, width - box_width)
                y1 = random.randint(0, height - box_height)
                x2 = x1 + box_width
                y2 = y1 + box_height
                draw.rectangle([x1, y1, x2, y2], fill='black')
            output_path = f"ImagesforResearch\\FIDConfirmationImages\\blocks\\{max_box_area}"
            if os.path.exists(output_path):
                pass
            else:
                os.makedirs(output_path)    
            img.save(os.path.join(output_path, originalImagePath))
        except:
            print(f"Image too small for {max_box_area}")

#Adds noise to image with the specified mean and standard deviation
def add_gaussian_noise(image, mean=0, std=1):
    from skimage.util import random_noise
    return torch.tensor(random_noise(image, mode="gaussian", mean=mean, var=std*std, clip=True)) 

#Adds Salt and Pepper noise to image
def add_salt_and_pepper_noise(image, noise_level=0.02):
    # Convert tensor to PIL Image
    image = transforms.ToPILImage()(image) #image should be a tensor here!
    width, height = image.size
    
    # Create a copy of the original image
    noisy_image = image.copy()
    
    num_pixels = width * height
    num_noisy_pixels = int(num_pixels * noise_level)
    
    for _ in range(num_noisy_pixels):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        
        if random.random() < 0.5:
            noisy_image.putpixel((x, y), (0, 0, 0))  # Salt noise
        else:
            noisy_image.putpixel((x, y), (255, 255, 255))  # Pepper noise
            
    return noisy_image

#Runs the process of generating augmented images. Calling this method will run all augmentations with the variables defined in the method below
def process(src_folder, method):
    image_files = [f for f in os.listdir(src_folder) if f.endswith('.jpg') or f.endswith('.png')]
    for item in image_files:
        image_path = os.path.join(src_folder, item)
        print(f"Working with {image_path} using {method}")
        image_tensor = prepareImage(image_path)
        if method == "blur":
            for i in range(1, 50, 2): #kernel_size needs to be an odd, positive integer
                mod_image = F.gaussian_blur(image_tensor, kernel_size=i)  # Adjust kernel_size as needed
                displayImage(mod_image, item, method,"kernal="+str(i)) #Do not pass the whole path, only the item name as we save to a new location (new path) but keep file name
        elif method == "saltandpepper":
            #noise_level should change by 0.05 each time
            for i in np.arange(start=0, stop=1+0.05, step=0.05):
                j = round(i, 2)
                mod_image = add_salt_and_pepper_noise(image_tensor, noise_level=j)  # Adjust noise_level as needed
                displayImage(mod_image, item, method, "noise="+str(j))
        elif method == "noise":
            stdev = 0.1 #Use this as anything beyond would overlap the new mean
            for mean in np.arange(start=0, stop=2+0.1, step=0.1):
                mean = round(mean, 2)
                mod_image = add_gaussian_noise(image_tensor, mean=mean, std=stdev)
                displayImage(mod_image, item, method, "mean="+str(mean)+"-stdev="+str(stdev))
        elif method == "blocks":
            mod_image = cover_with_random_boxes(image_path=image_path,originalImagePath = item )
        else:
            print("Method not supported")

#A helper function. The "src_folder" is the source images to augment.
def runProcesses():
    #This will take all images in src_folder and add the DA to them all and save to the relevant locations
    src_folder = "ImagesforResearch\\FIDConfirmationImages\\originalImages"
    #process(src_folder=src_folder, method="blocks")
    #return
    methods = ["blur", "noise", "saltandpepper", "blocks"]
    for method in methods:
        process(src_folder, method=method)


def calculateFID(image_folder, originalImages):
    #src_folder = "ImagesforResearch\\FIDConfirmationImages\\CelebA_Runs\\CelebA"
    FIDclass = FeatureExtraction.FeatureExtractionClass()
    return FIDclass.getFIDs(originalImages, image_folder)
    #for root, _, files in os.walk(src_folder):

def writeToCSV(filename,lists):
    with open(filename, mode = "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(lists)

def main():
    # ---------------
    #   Run this code to generate augmentations
    #runProcesses()
    #return
    # ---------------
    '''
    folders =   ["blur", "noise", "saltandpepper", "blocks"]
    for folder in folders:
        #folder = "blocks"
        image_folder = f"ImagesforResearch\\FIDConfirmationImages\\{folder}"
        print(image_folder)
        print("__________")
        csvData = []
        headers = ["Parameter", "FID"]
        csvData.append(headers)
        for root, _, files in os.walk(image_folder):
            if root == image_folder:
                #Skip main folder as we're just concerned with the subdirectories in the main folder
                continue
            print(root)
            x = root.rfind("\\") #extracts the parameter part of the path
            param = root[x+1:]
            src_folder = "ImagesforResearch\\FIDConfirmationImages\\originalImages"
            fid = calculateFID(root, originalImages=src_folder)
            print(root, "--->", fid)
            temp = [param, fid]
            csvData.append(temp)
        #return
        print(csvData)
        writeToCSV(f"MedFID\\ParameterFIDScores\\{folder}.csv", csvData)
    #calculateFIDs(image_folder=image_folder)
    '''
    folders =   ["VanillaRuns\\E1","VanillaRuns\\E2","VanillaRuns\\E3","VanillaRuns\\E4","VanillaRuns\\E5"]
    for folder in folders:
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        csvData = []
        for subfolder in subfolders:
            print(subfolder)
            headers = ["Parameter", "FID"]
            index = subfolder.rfind("\\")
            param = subfolder[index+1:]
            src_folder = "VanillaRuns\\original"
            fid = calculateFID(subfolder, originalImages=src_folder)
            print(subfolder, "--->", fid)
            temp = [param, fid]
            csvData.append(temp)
        print(csvData)
        writeToCSV(f"MedFIDAttempt2\\{folder}.csv", csvData)

if __name__ == "__main__":
    main()