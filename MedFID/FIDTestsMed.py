import os
import numpy as np
import csv
import FeatureExtraction
class FIDTests:

    def __init__(self):
        self.FIDclass = FeatureExtraction.FeatureExtractionClass()

    def calculateFID(self,image_folder, originalImages):
        #src_folder = "ImagesforResearch\\FIDConfirmationImages\\CelebA_Runs\\CelebA"
        
        return self.FIDclass.getFIDs(originalImages, image_folder)
        #for root, _, files in os.walk(src_folder):

    def writeToCSV(self,filename,lists):
        with open(filename, mode = "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(lists)

    def processFID(self):
        #runProcesses()
        #return
        
        folders = ["blur", "noise", "saltandpepper", "blocks"]
        for folder in folders:
            #folder = "blocks"
            image_folder = f"/mnt/lustre/users/schetty1/ImagesforResearch/FIDConfirmationImages/{folder}"
            print("Working in:", image_folder)
            csvData = []
            headers = ["Iteration", "FID"]
            csvData.append(headers)
            for root, _, files in os.walk(image_folder):
                if root == image_folder:
                    #Skip main folder as we're just concerned with the subdirectories in the main folder
                    continue
                print(root)
                x = root.rfind("/") #extracts the parameter part of the path
                param = root[x+1:]
                src_folder = "/mnt/lustre/users/schetty1/ImagesforResearch/FIDConfirmationImages/originalImages"
                fid = self.calculateFID(root, originalImages=src_folder)
                print(param)
                #fid = 0
                print(root, "--->", fid)
                temp = [param, fid]
                csvData.append(temp)
            #return
            print(csvData)
            self.writeToCSV(f"/home/schetty1/MedFIDAttempt2/ParameterFIDScores/FineTunedMed/{folder}.csv", csvData)
        #calculateFIDs(image_folder=image_folder)
        
def main():
    cls = FIDTests()
    cls.processFID()

if __name__ == "__main__":
    main()