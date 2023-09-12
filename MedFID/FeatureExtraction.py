import numpy as np
import numpy as np
import tensorflow as tf
import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import os
class FeatureExtractionClass:
    def __init__(self):
        labels = {'FootAnkle': 0, 'Knee': 1, 'XR_ELBOW': 2, 'XR_FOREARM': 3, 'XR_HAND': 4, 'XR_HUMERUS': 5, 'XR_SHOULDER': 6}
        newDict = {}
        for k in labels:
            newDict[labels[k]] = k
        labels = newDict
        
        model = keras.models.load_model("/home/schetty1/MedFIDAttempt2/FineTunedWeights.hdf5")
        model = keras.models.Model(model.input, model.layers[-2].output) 
        print(model.summary())
        self.model = model


    def getFeatures(self, img_path):
        from tensorflow.keras.utils import load_img, img_to_array #from keras.preprocessing import image
        img = load_img(img_path, target_size=(299, 299)) #Get to the right size
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x) 
        return preds

    def calculate_fid_numpy(self, act1, act2):
        import numpy
        from numpy import cov
        from numpy import trace
        from numpy import iscomplexobj
        
        from scipy.linalg import sqrtm
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    # _____________________________________
    # Calculator
    # _____________________________________

    def getFIDs(self, realFiles, fakeFiles):
        import numpy
        import os
        real_images = []
        folder_path = realFiles
        all_items = os.listdir(folder_path)
        for item in all_items:
            if os.path.isfile(os.path.join(folder_path, item)):
                real_images.append(self.getFeatures(os.path.join(folder_path, item)))
        real_images = numpy.concatenate(real_images, axis=0)
        print("--- real ---")
        print(real_images, "of type", type(real_images), "of size", real_images.shape)
        print("--- real ---")
        
        #fake items
        fake_images = []
        folder_path = fakeFiles 
        all_items = os.listdir(folder_path)
        for item in all_items:
            if os.path.isfile(os.path.join(folder_path, item)):
                fake_images.append(self.getFeatures(os.path.join(folder_path, item)))
        fake_images = numpy.concatenate(fake_images, axis=0)
        print("--- fake ---")
        print(fake_images, "of type", type(fake_images), "of size", fake_images.shape)
        print("--- fake ---")
        fid = self.calculate_fid_numpy(real_images, fake_images)
        return ('FID: %.3f' % fid)
    

def main():
    print(tf.__version__)
    print(keras.__version__)
    folder1 = "/mnt/lustre/users/schetty1/ImagesforResearch/Elbow/MOVEOUT"
    folder2 =  "/mnt/lustre/users/schetty1/GeneratedImages/StyleGAN2ADA/Elbow256/test"
    FIDclass = FeatureExtractionClass()
    print(FIDclass.getFIDs(realFiles=folder1, fakeFiles=folder2))

if __name__=="__main__":
    main()
