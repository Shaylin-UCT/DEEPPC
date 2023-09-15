import os
import numpy as np
import csv
import FeatureExtraction
class FIDTests:

    def __init__(self):
        self.FIDclass = FeatureExtraction.FeatureExtractionClass()

    def calculateFID(self,image_folder, originalImages):
        return self.FIDclass.getFIDs(originalImages, image_folder)

    def writeToCSV(self,filename,lists):
        with open(filename, mode = "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(lists)

    def processFID(self):
        folders = ["/mnt/lustre/users/schetty1/GeneratedImages/StyleGAN2ADA/Elbow256/Test2"]
        for folder in folders:
            image_folder = f"/mnt/lustre/users/schetty1/GeneratedImages/StyleGAN2ADA/Elbow256/{folder}"
            print("Working in:", image_folder)
            csvData = []
            headers = ["Set", "FID"]
            csvData.append(headers)
            for root, _, files in os.walk(image_folder):
                if root == image_folder:
                    #Skip main folder as we're just concerned with the subdirectories in the main folder
                    continue
                print(root)
                x = root.rfind("/") #extracts the parameter part of the path
                param = root[x+1:]
                src_folder = "/mnt/lustre/users/schetty1/GeneratedImages/SampleofOriginalImages"
                fid = self.calculateFID(root, originalImages=src_folder)
                print(param)
                #fid = 0
                print(root, "--->", fid)
                temp = [param, fid]
                csvData.append(temp)
            #return
            print(csvData)
            self.writeToCSV(f"/home/schetty1/MedFIDAttempt2/ParameterFIDScores/StyleGAN/{folder}.csv", csvData)
        #calculateFIDs(image_folder=image_folder)
        
def main():
    cls = FIDTests()
    cls.processFID()

if __name__ == "__main__":
    main()
