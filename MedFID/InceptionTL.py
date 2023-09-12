# ------------------------- Set up training -------------------------                  
steps_per_epoch = 50
epochs = 50
validation_steps = 10
# ------------------------- Set up training -------------------------

import numpy as np
np.random.seed(12)
import tensorflow as tf
import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import os
import matplotlib.pyplot as plt

# ----------------------------------------------------------
#           Prepare Dataset + print train images
# ----------------------------------------------------------

base_dir = "/mnt/lustre/users/schetty1/ImagesforResearch/MuraExtra2"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "valid")

#Scale and apply DA to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   horizontal_flip = True,
                                   preprocessing_function=preprocess_input
                                   )

image_size = 299

test_datagen = ImageDataGenerator(rescale = 1.0/255.)

#The bottom two will print out "Found x images belonging to y classes"
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20, 
                                                    class_mode = "categorical",
                                                    target_size = (image_size, image_size))

test_generator = test_datagen.flow_from_directory(test_dir,
                                                    batch_size = 20, 
                                                    class_mode = "categorical",
                                                    target_size = (image_size, image_size))

print(train_generator.class_indices, "of length:", len(train_generator.class_indices))
print(test_generator.class_indices, "of length:", len(test_generator.class_indices))

# ----------------------------------------------------------
#                 Import and adjust model
# ----------------------------------------------------------

pretrained_model = keras.models.load_model("/home/schetty1/MedFIDAttempt2/baseInception.h5")
print("loaded model")
#Test file to write output too
def myprint1(s):
    with open('/home/schetty1/MedFIDAttempt2/base.txt','a') as f:
        print(s, file=f)

pretrained_model.summary(print_fn=myprint1)


#Freeze all layers -> could also use "pretrained_model.trainable = False"
for layer in pretrained_model.layers:
    layer.trainable = False

#Add layers
#Use RMSprop optimizer (follows paper so this is good!) and set Lr to be 0.0001
from keras.optimizers import RMSprop
#Flatten the output layer to 1D
from keras import layers, Model
from keras.models import Sequential


add_model = Sequential()
add_model.add(pretrained_model)
def myprint2(s):
    with open('/home/schetty1/MedFIDAttempt2/sequential.txt','a') as f:
        print(s, file=f)

add_model.summary(print_fn=myprint2)
add_model.add(layers.GlobalAveragePooling2D())
add_model.add(layers.Dense(2048,activation="relu"))
add_model.add(layers.Dense(2048,activation="relu"))
add_model.add(layers.Dense(2048,activation="relu"))
add_model.add(layers.Dense(2048,activation="relu"))
add_model.add(layers.Dropout(0.2))
add_model.add(layers.Dense(7,activation="softmax")) 

def myprint2(s):
    with open('/home/schetty1/MedFIDAttempt2/sequential_updated.txt','a') as f:
        print(s, file=f)

add_model.summary(print_fn=myprint2)

model = add_model

model.compile(optimizer = RMSprop(learning_rate=0.0001), 
              loss = "categorical_crossentropy",  
              metrics = ["acc"])


# ----------------------------------------------------------
#                           Train
# ----------------------------------------------------------

weights_file_path = "/home/schetty1/MedFIDAttempt2/Clusterweights.hdf5" #Needs a hdf5 format
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
checkpoint = ModelCheckpoint(weights_file_path, monitor="acc", verbose=1, save_best_only=True, mode = "max")
early = EarlyStopping(monitor="acc", mode="max", patience=15) #See what these do
callbacks_list = [checkpoint, early]

#Create a Callback class that stops training once we have set accuracy:
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        stop_training_on = 0.90
        if(logs.get('acc')>stop_training_on):
            print("\nReached 90 accuracy so cancelling training!")
            self.model.stop_training = True
history = model.fit_generator(
    train_generator, 
    validation_data = test_generator,
    steps_per_epoch = steps_per_epoch, 
    epochs = epochs, 
    validation_steps = validation_steps, 
    verbose = 2, 
    callbacks=callbacks_list
)

results = model.evaluate(test_generator)
model.save("/home/schetty1/MedFID/clusterInceptionTest.h5")
print("results:", results)


# ----------------------------------------------------------
#                   Plot accuracy and loss
# ----------------------------------------------------------
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
print("!!!", epochs)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.savefig("/home/schetty1/MedFIDAttempt2/Accfigures.png")

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.savefig("/home/schetty1/MedFIDAttempt2/Lossfigures.png")

# ----------------------------------------------------------
#                     Make predictions
# ----------------------------------------------------------

print("-- Starting Predictions --")

def itemsInFolder(folder_path):
    items = os.listdir(folder_path)
    return items

from keras.preprocessing import image
folder_path = "/mnt/lustre/users/schetty1/ImagesforResearch/MuraExtra2/valid/XR_ELBOW" #test images -> do for ap and lat
predicts = []
do = True
if do == True:
    predicts = model.predict_generator(test_generator,
                                    verbose = True,
                                    workers=1)
    np.set_printoptions(threshold=np.inf)
    predicts = np.argmax(predicts, axis=-1) 
    label_index = {v: k for k,v in test_generator.class_indices.items()}
    predicts = [label_index[p] for p in predicts]
    

import pandas as pd

df = pd.DataFrame(columns=['fname', 'type']) #uses Pandas
df['fname'] = [os.path.basename(x) for x in test_generator.filenames]
df['type'] = predicts
df.to_csv("/home/schetty1/MedFIDAttempt2/InceptionTLResults.csv", index=False)
print("Done")