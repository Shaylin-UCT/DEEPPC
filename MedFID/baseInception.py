class baseInception:
    def getFeatures(self, img_path):
        import numpy as np
        import numpy as np
        import tensorflow as tf
        import keras
        from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
        import os
        print(keras.__version__)
        pretrained_model = keras.models.load_model("/home/schetty1/MedFIDAttempt2/baseInception.h5")
        #from tensorflow.keras.preprocessing.image import img_to_array
        #from keras.utils import load_img
        from tensorflow.keras.utils import load_img, img_to_array
        img = load_img(img_path, target_size=(299, 299)) #Get to the right size
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = pretrained_model.predict(x) #The weights when include_top is False else it is the predictions
        print(preds.shape)
        img_means=[preds[:,:,:,i].mean() for i in range(preds.shape[3]) ]
        print(img_means)
        #print("Predicted:", decode_predictions(preds, top=5)[0]) #Prints out top 2 predicted classes



def main():
    features = baseInception()
    features.getFeatures('/mnt/lustre/users/schetty1/ImagesforResearch/testerImages/Golden_Retriever_Dukedestiny01_drvd.jpg')

if __name__=="__main__":
    main()